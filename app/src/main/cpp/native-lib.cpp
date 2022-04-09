#include <jni.h>
#include <string>
#include "EMD/EmpiricalModeDecomposition.h"

extern "C"
JNIEXPORT jdoubleArray JNICALL
Java_com_example_mytest_MainActivity_dataProcess(JNIEnv *env, jobject thiz, jdoubleArray input) {
    // TODO: implement dataProcess()
    jsize size= env->GetArrayLength(input);
    jdouble *doubleArray = env->GetDoubleArrayElements(input,JNI_FALSE);


}