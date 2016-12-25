#ifndef MY_SVM_H
#define MY_SVM_H

//继承自CvSVM的类，因为生成setSVMDetector()中用到的检测子参数时，需要用到训练好的SVM的decision_func参数，
//但通过查看CvSVM源码可知decision_func参数是protected类型变量，无法直接访问到，只能继承之后通过函数访问
class MySVM : public CvSVM
{
public:
    //获得SVM的决策函数中的alpha数组
    double * get_alpha_vector()
    {
        return this->decision_func->alpha;
    }

    //获得SVM的决策函数中的rho参数,即偏移量
    float get_rho()
    {
        return this->decision_func->rho;
    }
};

#endif
