#include <iostream> 
#include <thread>   // 线程库
#include <vector>   
#include <random>   // 随机数生成库(mt19937随机数生成器）
#include <mutex>    // 互斥锁库
#include <chrono>   // 包含时间测量库，用于计算程序运行时间
using namespace std;

mutex mtx; // 定义一个全局互斥锁，用于保护共享变量的访问
long long totalPointsInCircle = 0; // 定义一个全局变量

void monteCarloPi(long long pointsPerThread, mt19937& rng)
{
    long long pointsInCircle = 0; 

    
    uniform_real_distribution<double> dist(-1.0, 1.0);

    
    for (long long i = 0; i < pointsPerThread; ++i)
    {
        double x = dist(rng); 
        double y = dist(rng); 

        
        if (x * x + y * y <= 1.0)
        {
            pointsInCircle++;
        }
    }

    lock_guard<mutex> lock(mtx); // 使用互斥锁保护全局变量访问
    totalPointsInCircle += pointsInCircle; // 将当前线程结果累加到全局计数器
}

int main()
{
    int numThreads;
    long long pointsPerThread;

    
    cout << "Enter the number of threads: ";
    cin >> numThreads;

    

    while(numThreads>0)
    {
        cout << "Enter the number of points per thread: ";
        cin >> pointsPerThread;

        if (pointsPerThread <= 0)
        {
            cerr << "Error: Number of points per thread must be positive." << endl;
            return 1;
        }
        long long totalPoints = numThreads * pointsPerThread;

        auto start = chrono::high_resolution_clock::now();
        random_device rd;
        mt19937 rng(rd());

      
        vector<thread> threads;
        for (int i = 0; i < numThreads; ++i)
        {
            threads.emplace_back(monteCarloPi, pointsPerThread, ref(rng));
        }


        for (auto& t : threads)
        {
            t.join(); // join() 阻塞主线程，直到该线程执行完毕
        }

        double pi = 4.0 * totalPointsInCircle / totalPoints;


        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;

        cout << "Estimated value of π: " << pi << endl;
        cout << "Time taken: " << duration.count() << " seconds" << endl;
        totalPointsInCircle = 0;

        cout << "\nEnter the number threads: ";
        cin >> numThreads;
    }

    if (numThreads <= 0)
    {
        cerr << "Error: Number of threads must be positive." << endl;
        cin.ignore((numeric_limits<streamsize>::max)(), '\n'); // 清空缓冲区
        return 1; 
    }

}