/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 6.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QWidget *widget;
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *verticalLayout;
    QLabel *label;
    QLineEdit *plainTextEdit;
    QLabel *label_2;
    QLineEdit *cipherTextEdit;
    QLabel *label_3;
    QLineEdit *keyTextEdit;
    QVBoxLayout *verticalLayout_2;
    QPushButton *encryptButton;
    QPushButton *decryptButton;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(800, 600);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName("centralwidget");
        widget = new QWidget(centralwidget);
        widget->setObjectName("widget");
        widget->setGeometry(QRect(100, 44, 481, 117));
        horizontalLayout = new QHBoxLayout(widget);
        horizontalLayout->setObjectName("horizontalLayout");
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName("verticalLayout");
        label = new QLabel(widget);
        label->setObjectName("label");

        verticalLayout->addWidget(label);

        plainTextEdit = new QLineEdit(widget);
        plainTextEdit->setObjectName("plainTextEdit");

        verticalLayout->addWidget(plainTextEdit);

        label_2 = new QLabel(widget);
        label_2->setObjectName("label_2");

        verticalLayout->addWidget(label_2);

        cipherTextEdit = new QLineEdit(widget);
        cipherTextEdit->setObjectName("cipherTextEdit");

        verticalLayout->addWidget(cipherTextEdit);

        label_3 = new QLabel(widget);
        label_3->setObjectName("label_3");

        verticalLayout->addWidget(label_3);

        keyTextEdit = new QLineEdit(widget);
        keyTextEdit->setObjectName("keyTextEdit");

        verticalLayout->addWidget(keyTextEdit);


        horizontalLayout->addLayout(verticalLayout);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName("verticalLayout_2");
        encryptButton = new QPushButton(widget);
        encryptButton->setObjectName("encryptButton");

        verticalLayout_2->addWidget(encryptButton);

        decryptButton = new QPushButton(widget);
        decryptButton->setObjectName("decryptButton");

        verticalLayout_2->addWidget(decryptButton);


        horizontalLayout->addLayout(verticalLayout_2);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName("menubar");
        menubar->setGeometry(QRect(0, 0, 800, 18));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName("statusbar");
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        label->setText(QCoreApplication::translate("MainWindow", "\350\257\267\345\234\250\344\270\213\346\226\271\350\276\223\345\205\245\345\276\205\345\212\240\345\257\206\347\232\204\346\230\216\346\226\207 / \350\247\243\345\257\206\345\220\216\347\232\204\346\230\216\346\226\207\345\260\206\346\230\276\347\244\272\345\234\250\344\270\213\346\226\271\357\274\232", nullptr));
        label_2->setText(QCoreApplication::translate("MainWindow", "\345\212\240\345\257\206\345\220\216\347\232\204\345\257\206\346\226\207\345\260\206\346\230\276\347\244\272\345\234\250\344\270\213\346\226\271 / \350\257\267\345\234\250\344\270\213\346\226\271\350\276\223\345\205\245\345\276\205\350\247\243\345\257\206\347\232\204\345\257\206\346\226\207\357\274\232", nullptr));
        label_3->setText(QCoreApplication::translate("MainWindow", "\350\257\267\345\234\250\344\270\213\346\226\271\350\276\223\345\205\24516\344\275\21516\350\277\233\345\210\266\345\257\206\351\222\245\357\274\232", nullptr));
        encryptButton->setText(QCoreApplication::translate("MainWindow", "\345\212\240\345\257\206", nullptr));
        decryptButton->setText(QCoreApplication::translate("MainWindow", "\350\247\243\345\257\206", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
