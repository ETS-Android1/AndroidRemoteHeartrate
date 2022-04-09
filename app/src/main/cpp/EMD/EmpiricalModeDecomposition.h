//
// Created by c4940 on 2021/10/21.
//
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifndef MYTEST_EMPIRICALMODEDECOMPOSITION_H
#define MYTEST_EMPIRICALMODEDECOMPOSITION_H
#define cnew(type, size) ((type*) malloc((size) * sizeof(type)))
#define cdelete(ptr) free(ptr)

typedef struct {
    int iterations, order, locality;
    int *minPoints, *maxPoints;
    float *min, *max, **imfs, *residue;
    int size, minSize, maxSize;
} emdData;

void emdSetup(emdData* emd, int order, int iterations, int locality);
void emdResize(emdData* emd, int size);
void emdCreate(emdData* emd, int size, int order, int iterations, int locality);
void emdClear(emdData* emd);
void emdDecompose(emdData* emd, const float* signal);
void emdMakeExtrema(emdData* emd, const float* curImf);
void emdInterpolate(emdData* emd, const float* in, float* out, int* points, int pointsSize);
void emdUpdateImf(emdData* emd, float* imf);
void emdMakeResidue(emdData* emd, const float* cur);
int mirrorIndex(int i, int size);
#endif //MYTEST_EMPIRICALMODEDECOMPOSITION_H
