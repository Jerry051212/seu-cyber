#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <dirent.h>     // 用于Linux/macOS下遍历目录
#include <sys/stat.h>
#include <utility>      // std::pair
#include <ctime>        // srand(time(nullptr))

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"  // 单文件PNG/JPG加载库
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// 常量定义
const int WIDTH = 32;                  // ORL人脸图像宽度
const int HEIGHT = 32;                 // ORL人脸图像高度
const int PIXELS = WIDTH * HEIGHT;     // 每张图的总像素数 = 1024
const int NUM_SUBJECTS = 40;           // 总共有40个人
const int TRAIN_PER_SUBJECT = 7;       // 每人取前7张作为训练集
const int TOTAL_TRAIN = NUM_SUBJECTS * TRAIN_PER_SUBJECT;  // 训练样本总数=280

// 简单向量类
class Vector {
public:
    std::vector<double> data;  // 实际存储数据的容器
    int size;                  // 向量长度

    Vector(int n = 0) : size(n), data(n, 0.0) {}  // 构造函数，全部初始化为0
    Vector(const std::vector<double>& v) : data(v), size((int)v.size()) {}

    // 重载[]操作符，便于访问元素
    double& operator[](int i) { return data[i]; }
    const double& operator[](int i) const { return data[i]; }

    // 向量加法
    Vector operator+(const Vector& other) const {
        Vector res(size);
        for (int i = 0; i < size; ++i) res[i] = data[i] + other[i];
        return res;
    }

    // 向量减法
    Vector operator-(const Vector& other) const {
        Vector res(size);
        for (int i = 0; i < size; ++i) res[i] = data[i] - other[i];
        return res;
    }

    // 数乘
    Vector operator*(double scalar) const {
        Vector res(size);
        for (int i = 0; i < size; ++i) res[i] = data[i] * scalar;
        return res;
    }

    // 点积（内积）
    double dot(const Vector& other) const {
        double sum = 0.0;
        for (int i = 0; i < size; ++i) sum += data[i] * other[i];
        return sum;
    }

    // L2范数（欧氏长度）
    double norm() const { return std::sqrt(dot(*this)); }

    // 单位化（归一化到长度为1）
    Vector normalize() const {
        double n = norm();
        if (n < 1e-10) return Vector(size);  // 防止除以0
        return *this * (1.0 / n);
    }
};

// 简单矩阵类 矩阵-向量乘、矩阵-矩阵乘、转置
class Matrix {
public:
    std::vector<Vector> rows;  // 用向量向量存储每一行
    int rows_num, cols_num;    // 行数和列数

    Matrix(int r = 0, int c = 0) : rows_num(r), cols_num(c), rows(r, Vector(c)) {}

    Vector& operator[](int i) { return rows[i]; }
    const Vector& operator[](int i) const { return rows[i]; }

    // 转置
    Matrix transpose() const {
        Matrix t(cols_num, rows_num);
        for (int i = 0; i < rows_num; ++i)
            for (int j = 0; j < cols_num; ++j)
                t[j][i] = rows[i][j];
        return t;
    }

    // 矩阵乘矩阵
    Matrix operator*(const Matrix& other) const {
        Matrix res(rows_num, other.cols_num);
        for (int i = 0; i < rows_num; ++i)
            for (int j = 0; j < other.cols_num; ++j)
                for (int k = 0; k < cols_num; ++k)
                    res[i][j] += rows[i][k] * other[k][j];
        return res;
    }

    // 矩阵乘向量
    Vector operator*(const Vector& vec) const {
        Vector res(rows_num);
        for (int i = 0; i < rows_num; ++i)
            res[i] = rows[i].dot(vec);
        return res;
    }
};

// 幂迭代法求最大特征值和特征向量
// PCA需要协方差矩阵的特征向量，但协方差矩阵是1024×1024太大，
// 我们采用“低秩技巧”：先求小矩阵L=A^T A (280×280)的特征向量，再转换。
// 这里用幂迭代法逐个求L的最大特征对，然后通过deflation消去已求部分。
std::pair<double, Vector> power_iteration(const Matrix& A, int max_iter = 1000, double tol = 1e-8) {
    // 随机初始化一个向量
    Vector v(A.cols_num);
    for (int i = 0; i < A.cols_num; ++i) v[i] = rand() % 100 + 1.0;
    v = v.normalize();

    double lambda_prev = 0.0;
    for (int iter = 0; iter < max_iter; ++iter) {
        Vector u = A * v;              // A*v
        double lambda = u.dot(v);      // Rayleigh商作为特征值估计
        v = u.normalize();             // 归一化得到下一个迭代向量

        // 收敛判断
        if (std::abs(lambda - lambda_prev) < tol) break;
        lambda_prev = lambda;
    }
    return { lambda_prev, v };  // 返回特征值和对应的特征向量
}

// 加载单张灰度图像并展平为向量
Vector load_image_to_vector(const std::string& path) {
    int w, h, channels;
    // 强制转为单通道灰度图
    unsigned char* img = stbi_load(path.c_str(), &w, &h, &channels, 1);
    if (!img || w != WIDTH || h != HEIGHT) {
        std::cerr << "加载失败或尺寸不对: " << path << std::endl;
        return Vector(PIXELS);  // 返回空向量表示错误
    }

    Vector vec(PIXELS);
    // 像素值归一化到[0,1]
    for (int i = 0; i < PIXELS; ++i) {
        vec[i] = static_cast<double>(img[i]) / 255.0;
    }
    stbi_image_free(img);
    return vec;
}

// 获取某个subject的所有图像路径
// 按命名规则subjectXX_YY.png查找，并按文件名自然顺序排序（确保01在最前）
std::vector<std::string> get_subject_images(const std::string& dir, int subject_id) {
    std::vector<std::string> files;
    DIR* dp = opendir(dir.c_str());
    if (!dp) return files;

    char subj_str[20];
    sprintf(subj_str, "subject%02d", subject_id);  // 如 "subject01"

    struct dirent* entry;
    while ((entry = readdir(dp)) != nullptr) {
        std::string name = entry->d_name;
        // 判断文件名是否以 "subjectXX" 开头且以 ".png" 结尾
        if (name.rfind(subj_str, 0) == 0 &&       // rfind(...,0) 表示从开头匹配
            name.size() >= 4 &&
            name.substr(name.size() - 4) == ".png") {
            files.push_back(dir + "/" + name);
        }
    }
    closedir(dp);

    std::sort(files.begin(), files.end());  // 排序保证 01,02,...10 的顺序
    return files;
}

int main() {
    srand(time(nullptr));  // 为幂迭代随机初始化种子

    std::string train_dir = "./orl_faces";  // 训练集目录

    std::cout << "开始加载训练集..." << std::endl;

    // 加载训练图像
    std::vector<Vector> train_images;  // 所有训练图像的向量形式（280个1024维向量）
    std::vector<int> labels;           // 对应每个训练样本的subject编号（1~40）

    for (int subj = 1; subj <= NUM_SUBJECTS; ++subj) {
        auto files = get_subject_images(train_dir, subj);
        if (files.size() < TRAIN_PER_SUBJECT) {
            std::cerr << "Subject " << subj << " 图像不足7张！" << std::endl;
            return -1;
        }

        // 只取前7张作为训练
        for (int i = 0; i < TRAIN_PER_SUBJECT; ++i) {
            Vector img = load_image_to_vector(files[i]);
            if (img.size == 0) return -1;
            train_images.push_back(img);
            labels.push_back(subj);
        }
    }

    std::cout << "加载完成，共 " << train_images.size() << " 张训练图像。" << std::endl;

    // 计算平均脸（mean face）
    Vector mean_face(PIXELS);
    for (const auto& img : train_images) mean_face = mean_face + img;
    mean_face = mean_face * (1.0 / TOTAL_TRAIN);  // 求平均

    // 中心化（减去平均脸）
    // PCA的核心：去均值后数据才在原点附近
    std::vector<Vector> centered;
    for (const auto& img : train_images) {
        centered.push_back(img - mean_face);
    }

    // 构造小协方差矩阵 L = A^T * A
    // A 是 PIXELS × TOTAL_TRAIN 的矩阵（行：像素，列：样本）
    // 直接计算 A*A^T 会得到 1024×1024 的巨大矩阵，内存和计算都吃不消
    // 所以我们计算 L = A^T * A，只有 280×280，大小可接受
    Matrix L(TOTAL_TRAIN, TOTAL_TRAIN);
    for (int i = 0; i < TOTAL_TRAIN; ++i) {
        for (int j = i; j < TOTAL_TRAIN; ++j) {  // 对称矩阵，只算上三角
            double val = centered[i].dot(centered[j]);  // 第i和第j样本的内积
            L[i][j] = val;
            L[j][i] = val;
        }
    }

    // 用幂迭代+deflation求L的前K个特征向量
    const int K = 50;  // 选取的主成分数量（特征脸数量）
    std::vector<Vector> eigen_vectors_low;  // L的特征向量（每个280维）
    std::vector<double> eigen_values;

    Matrix L_copy = L;  // 复制一份用于deflation破坏
    for (int k = 0; k < K; ++k) {
        std::pair<double, Vector> result = power_iteration(L_copy);
        double lambda = result.first;
        Vector v = result.second;

        if (lambda < 1e-8) break;  // 特征值太小，后面的可以忽略

        eigen_values.push_back(lambda);
        eigen_vectors_low.push_back(v);

        // Deflation：从矩阵中减去当前秩1成分，使下次迭代得到次大特征值
        for (int i = 0; i < TOTAL_TRAIN; ++i) {
            for (int j = 0; j < TOTAL_TRAIN; ++j) {
                L_copy[i][j] -= lambda * v[i] * v[j];
            }
        }
    }

    // 转换得到原始空间的特征脸（eigenfaces）
    // 数学原理：原始协方差矩阵的特征向量 u = A * v_low（v_low是L的特征向量）
    std::vector<Vector> eigenfaces;  // 真正的“特征脸”，每个是1024维
    for (const auto& v_low : eigen_vectors_low) {
        Vector face(PIXELS);
        for (int i = 0; i < TOTAL_TRAIN; ++i) {
            face = face + centered[i] * v_low[i];  // 线性组合
        }
        face = face.normalize();  // 归一化，便于后续比较
        eigenfaces.push_back(face);
    }

    int actual_K = eigenfaces.size();
    std::cout << "PCA完成，提取了 " << actual_K << " 个特征脸。" << std::endl;

    // 将所有训练样本投影到特征脸子空间
    // 得到低维表示（每个样本从1024维降到actual_K维）
    std::vector<std::vector<double>> train_proj(TOTAL_TRAIN, std::vector<double>(actual_K));
    for (int i = 0; i < TOTAL_TRAIN; ++i) {
        for (int j = 0; j < actual_K; ++j) {
            train_proj[i][j] = centered[i].dot(eigenfaces[j]);  // 投影系数
        }
    }

    // 交互式识别循环
    while (true) {
        std::cout << "\n请输入测试图像路径（输入 quit 退出）: ";
        std::string test_path;
        std::getline(std::cin, test_path);

        if (test_path == "quit" || test_path == "q") break;

        Vector test_img = load_image_to_vector(test_path);
        if (test_img.size == 0) continue;

        // 测试图像也要减平均脸并投影
        Vector test_centered = test_img - mean_face;
        std::vector<double> test_proj(actual_K);
        for (int j = 0; j < actual_K; ++j) {
            test_proj[j] = test_centered.dot(eigenfaces[j]);
        }

        // 在低维空间用最近邻（欧氏距离）分类
        double min_dist = 1e100;
        int best_subject = -1;

        for (int i = 0; i < TOTAL_TRAIN; ++i) {
            double dist = 0.0;
            for (int j = 0; j < actual_K; ++j) {
                double diff = train_proj[i][j] - test_proj[j];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);

            if (dist < min_dist) {
                min_dist = dist;
                best_subject = labels[i];
            }
        }

        std::cout << "识别结果：最相似的人脸是 subject"
            << (best_subject < 10 ? "0" : "") << best_subject
            << " （距离: " << min_dist << "）" << std::endl;
    }

    return 0;
}