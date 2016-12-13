/* compile with openMP flags in gcc, link with BLAS
 * For Unix architecture:
 * mex -largeArrayDims CC=gcc CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" plowMex.c -lmwblas 
 * For Windows:
 * mex -largeArrayDims kmeansMex.c <absolute_path_to_/libmwlapack.lib>
 *
 * Corresponding Matlab version is plowFast.m
 */

#include <math.h>
#include "mex.h"
#include "matrix.h"
#include <omp.h>
#include "blas.h"

/*#if !defined(_WIN32)
#define dsyrk dsyrk_
#define dgemm dgemm_
#define daxpy daxpy_
#define dgemv dgemv_
#endif*/


#define	Y_IN	prhs[0]
#define	I_IN	prhs[1]
#define IMGH    prhs[2]
#define IMGW    prhs[3]
#define ZMN_IN  prhs[4]
#define V_IN   prhs[5] 
#define D_IN   prhs[6]
#define REFIDX_IN  prhs[7]
#define sig2_IN prhs[8]
#define TH_IN   prhs[9]
#define PREP_IN prhs[10]


/* Output Arguments */

#define	Z_OUT	plhs[0]
#define Ce_OUT   plhs[1]
/*#define WTS_OUT  plhs[2]*/

/* #define DIST_OUT plhs[2]*/

#define sqr(a) (a)*(a)
#define max(a,b) ((a>=b)?a:b)
#define min(a,b) ((a<=b)?a:b)
#define FALSE 0
#define TRUE 1

double ddot(mwSignedIndex *, double *, mwSignedIndex *, double *, mwSignedIndex *);


void getNNDist(double *, int, int, double *, int, int, int, int, double, int *, double *, int *);


void calcErrorMat(int, double *, double *, double, double *, double *); 
int findInsIdx(double *, double, int, int);

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] )
{ 
    mwSize mrows, ncols, midx, nidx, csz, dims[2];
    double *refIdx,*Z, *Ce, *U, *D, *V, *wyj, *yi, *yj, *Y, *I, *zmn, *wts, *E, *nndist2, *wts_out, *zi;
    int  *p2, *idx_nn, *idx;
    int ksz, i,j, d, wrad, el, rad, p=10, ssz=30, tid, nThreads;
    double sig2, thresh, h2, wSum, wSumNeg, mul=0.0;
    mwSignedIndex d2, incx=1, ncols2, ksz2;
    int prep, ro, co, r, c, imgW, imgH;
        
    
    char *tran="N";
    double alpha = 1.0;
    
    
    
     /* Get dimensions of MXN matrix*/
   mrows = mxGetM(Y_IN); /*Row size - size of each vector in Y*/
   ncols = mxGetN(Y_IN); /* Column size - number of vectors*/
   midx = mxGetM(REFIDX_IN);
   nidx = mxGetN(REFIDX_IN);
   csz = midx*nidx;
   
                              
   /* Assign pointers to parameters */
   Y = (double *)(mxGetPr(Y_IN));
   I = (double *)(mxGetPr(I_IN));
   imgH = (int)(mxGetScalar(IMGH));
   imgW = (int)(mxGetScalar(IMGW));
   zmn = (double *)(mxGetPr(ZMN_IN));
   V = (double *)(mxGetPr(V_IN));
   D = (double *)(mxGetPr(D_IN));
   refIdx = (double *)(mxGetPr(REFIDX_IN));
   
   sig2 = (double)(mxGetScalar(sig2_IN)); 

   prep = (int)(mxGetScalar(PREP_IN));
   
   d = (int)mrows;
   ksz = sqrt(d);
   rad = (ksz-1)/2;
   d2 = (mwSignedIndex)d;
   ksz2 = (mwSignedIndex)ksz;
   wrad = ssz/2;
   thresh = (double)(mxGetScalar(TH_IN)); 
   
   
 /*printf("size %dx%d\n",mrows,ncols);
   for(i=0;i<10;i++)
       printf("%0.2f ",*(Y + i));
   printf("\n");
   return;*/
   
   
   
   Z_OUT = mxCreateDoubleMatrix(1,csz*d,mxREAL);
   Z = (double *)(mxGetPr(Z_OUT));
   
   Ce_OUT = mxCreateDoubleMatrix(1,csz*d,mxREAL);
   Ce = (double *)(mxGetPr(Ce_OUT));
   
  /* WTS_OUT = mxCreateDoubleMatrix(1,csz*p,mxREAL);
   wts_out = (double *)(mxGetPr(WTS_OUT));*/
   
   h2 = 1.75*sig2*((double)d);
   /*printf("\nProcessing with noise std dev %0.2f\n", sqrt(sig2));  */
   
   /* Here goes the actual PLOW part */
 /*   #pragma omp parallel shared(Y,I,V,Z,D,Ridx,Cidx,mrows,ncols,nThreads,wrad,thresh,p,d2) private(E, wts, tid,i,j, el, wyj, yj, p2,idx_nn,nndist2)
   {
       tid = omp_get_thread_num();
        if (tid == 0)
            nThreads = omp_get_num_threads();*/
   
       wts = (double*)malloc(p*sizeof(double));
   
       E = (double *)malloc(sqr(d)*sizeof(double));
       wyj = (double *)malloc(d*sizeof(double));
       yj = (double *)malloc(d*sizeof(double));
       idx_nn = (int *)malloc(p*sizeof(int));
       nndist2 = (double *)malloc(p*sizeof(double));
       p2 = (int *)malloc(sizeof(int));
       

   /*printf("entering main loop\n");*/
       
    /*   #pragma omp for nowait /* Run in parallel */   
       for(i=0; i<csz; i++){

           /* Reset wyj to all zeroes */
           zi = Z + i*d;
           
           for (el=0; el<d; el++){
               *(wyj + el) = 0.0;
               *(zi+el) = *(zmn + el);
           }


           /* Form the search window, get top p NN distances, and their indices*/
           
         /* printf("Get neighbor distances...");*/
           getNNDist(Y, imgH, imgW, yj, (int)(*(refIdx + i)), ksz, wrad, p, thresh, idx_nn, nndist2, p2);
           /* printf("Done.\n");
           printf("Returned %d matches.\n",*p2);*/

           /* Compute weights from distances */
           
           mul = ((prep==1) ? ((double)*p2)/sig2: 1.0/sig2); /* Controlled smoothing in prepocessing step */
           
           
           /*printf("Computing weighted patch ... ");*/
           for (j=0, wSum=0; j < *p2; j++){

               *(wts + j) = mul*(exp(-(*(nndist2 + j))/h2));
          
                /* wyj = wyj + wts[j] * (yj - zmn) */
               
                /* Column major, since we need to send data back to matlab */
                
                 /* wyj = wyj + wts[j] * yj */
                daxpy(&d2, wts+j, I + (*(idx_nn+j))*d, &incx, wyj, &incx);
               /*  printf("(%d, %0.3f) ", *(idx_nn + j), *(nndist2 + j));*/
                          
                wSum += *(wts+j);
           }
          /* printf("\n");
            printf(" Stage 2.... ");*/
           
           /* wyj = wyj - wSum*zmn */
           wSumNeg = -1.0*wSum;
           daxpy(&d2, &wSumNeg, zmn, &incx, wyj, &incx);
          /* printf("Done.\nComputing Error Mat..... ");*/

           /* inv(inv(Cz) + wSum) */
           calcErrorMat(d,V,D,wSum,E,Ce+i*d); 
          /* printf("Done.\nComputing final estimate....");*/

           /* Z(:,i) = zmn + E*wyj */
           dgemv(tran, &d2, &d2, &alpha, E, &d2, wyj, &incx, &alpha, zi, &incx);
            /*printf("Done\n.");*/
   }
   
   
   free(wyj);
   free(yj);
   free(E); 
   free(wts);
   free(nndist2);
   free(idx_nn);
   free(p2);
/*   } /* End of parallel block */
   
}


                                
/* Worth the overhead for large nearest neighbor searches
 If two values are equal, lower index goes first */
int findInsIdx(double *nndist2, double D, int loIdx, int upIdx)
{
      int minc, maxc, minr, maxr, cen;
           
            
      do{
          cen=loIdx + (upIdx-loIdx)/2;
          
          if(nndist2[loIdx]>D) return loIdx; /* d less than first element */
          else if(nndist2[upIdx]<=D) return upIdx+1; /* d greater than last */
          else if(nndist2[cen]>D)  /* d less than center, use left half*/
              upIdx = cen-1; 
          else               /* d >= center, use right half */
              loIdx = cen+1;
      } while(loIdx <= upIdx);
      
      return loIdx;
}




/* E  = V*diag(D + wSum)*V'; Ce = diag(E); 
 V : dxd, D : dx1, E : dxd, Ce : dx1   */
void calcErrorMat(int d, double *V, double *D, double wSum, double *E, double *Ce)
{
    double *VD; /* *VD,*/
    int elr, elc, ell, ri, cj, ricj;
    double D2;
    int tid, nthreads, chunk = 10; /* Chunk size for each thread */
    /* For BLAS */
    char *tran = "N", *tran2 = "T";
    char *uplo = "L";
    double alpha = 1.0, beta = 0.0;
    mwSignedIndex d2, dsq, incr=1;
    d2 = (mwSignedIndex)d;
    dsq = sqr(d2);
            
    VD = (double *)malloc(sqr(d)*sizeof(double)); /* to hold V*D */
    dcopy(&dsq, V, &incr, VD, &incr); /* Copy V into VD */
    
      for (elc=0; elc < d; elc++){ 
         D2 = (*(D+elc))/(1.0 + (*(D+elc))*wSum) - 1; /* -1 to make V(:,elc).*D2 = V(:,elc).*(D2-1) + V(:,elc); */
         /* point-wise mul of rows, so skip of d */
         daxpy(&d2, &D2, (V+elc), &d2, (VD + elc), &d2); 
     }
           
     
    
    /* Now compute VD*V' */
    /* Ideally dsyrk should be faster but for smaller matrices, copying the diagonal form takes more time*/   
    /*dsyrk(uplo,tran, &d2, &d2, &alpha, VD, &d2, &beta, E, &d2);*/
    /* Fortran reads column-wise, so send in transposes */
     dgemm(tran2, tran, &d2, &d2, &d2, &alpha, VD, &d2, V, &d2, &beta, E, &d2);
     
    /* Now  populate Ce */
    dsq = d2 + 1;
    dcopy(&d2, E, &dsq, Ce, &incr); 
    
    free(VD);
}
    
  



void getNNDist(double *Y,  int imgH, int imgW, double *yj, int refIdx, int ksz, int wrad, int p, double thresh,int *idx_nn, double *nndist2, int *p2)
{
    /* ridx and cidx are assumed to be from original image, not with the paddings 
     * but Y is assumed to be the padded image */
    int i, j, elr, elc, l, s, ri, ci, h, w, lm, sn, rad = (ksz-1)/2, d = sqr(ksz);
    int idx,minc, maxc, minr, maxr, ridx, cidx, jidx, acceptFlag=TRUE;
   
    double D, *yi, diff[1], thrMax, negOne=-1.0;
    /*double *yj = (double *)malloc(ksz*sizeof(double));*/
    mwSignedIndex d2 = (mwSignedIndex)d, incr=1;
    
    cidx = (int)(refIdx/imgH);
    ridx = (int)(refIdx % imgH);
    
    /*printf("Index = [%d,%d]\n",ridx,cidx);*/
    /* Form list of indices for window */
    minr = max(ridx - wrad, 0);
    minc = max(cidx - wrad, 0);
    maxc = min(cidx + wrad, imgW);
    maxr = min(ridx + wrad, imgH);
   
    
    for (l=0; l<p; l++){
        *(nndist2+l) = thresh;
    }
    
    *p2 = -1;
    
    
    /* Search for top p similar patches within a search window */
   
    yi = Y + refIdx*d;
    
    
     /* Matlab sends patches in row-major form */
    for(s=minc; s<maxc; s++){   
       for(l=minr, jidx = (s*imgH + minr)*d; l<maxr; l++, jidx+= d){   
           acceptFlag=TRUE;
           thrMax = *(nndist2+p-1);
           
                     
           dcopy(&d2, yi, &incr, yj, &incr); /* yj = yi */
           daxpy(&d2, &negOne, Y + jidx, &incr, yj, &incr); /* yj = yi - yj */     
           diff[0] = ddot(&d2, yj, &incr, yj, &incr); /* diff = sum(yj.^2) */
           
           if(diff[0] >= thrMax) { /* distance already greater, do not proceed with this patch */
                acceptFlag = FALSE;
                continue;
           }
              
           
          if(acceptFlag){
                
                (*p2)++;
                if(*p2 >= p) /* p2 is always less than p */
                    *p2 = p-1; 
                             
                /* Find where to insert the index */
                idx = findInsIdx(nndist2,diff[0],0,*p2);
                /*printf("\nInsert into %d\n",idx);*/
                
                elr = min(*p2+1,p-1); 
                while(elr>idx){
                        
                    *(nndist2 + elr) = *(nndist2 + elr - 1);
                    *(idx_nn + elr) = *(idx_nn + elr - 1);
                    elr--;
                     /*printf("%d:%f ", elr, *(nndist2 + elr));  */
                }
                
                    
                *(nndist2 + idx) = diff[0]; 
           /*     printf("%u:%f\n",nndist2+idx, *(nndist2+idx));*/
                *(idx_nn + idx) = s*imgH + l;
          }
       }
        
    }
    
    (*p2) += 1; /* convert from index of last element to number of elements */ 
   
}
    


    



