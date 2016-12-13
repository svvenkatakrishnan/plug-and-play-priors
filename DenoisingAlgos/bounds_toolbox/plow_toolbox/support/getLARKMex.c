/* Compile with Blas and lapack
 * mex -O -largeArrayDims CC=gcc CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" getLARKMex.c -lmwblas 
 */

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "mex.h"
#include "matrix.h"
#include <omp.h>
#include "blas.h"
#include "lapack.h"
#include <string.h>

#define	IMG_IN	prhs[0]
#define WSZ_IN  prhs[1]
#define H_IN    prhs[2]

#define	W_OUT	plhs[0]


#define	max(A, B)	((A) > (B) ? (A) : (B))
#define	min(A, B)	((A) < (B) ? (A) : (B))

/*extern void dgesvd( char*, char*, int*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, int*);*/
double ddot(mwSignedIndex *, double *, mwSignedIndex *, double *, mwSignedIndex *);


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] )
{ 
   mwSize mrows, ncols, midx, nidx, dims[2];
   int d, r, c, l, gradHeight, gradWidth, height, width, rad,  rptr, r2, c2, Cidx; 
   double *zx, *zy, *img, *W, One =1.0, alpha = 0.5, h2, h, *G, *yp, wsum;
   mwSignedIndex inr=1, d2, two=2, info, gw, gh, wsz, five=5, twenty5 = 25;
   double *dx, *dy, tmp[4], S[2], ss, *u, *vt, *s, wkopt, *work, *C11, *C12, *C22;
   double Ax[] = {0.0, -0.0003,  -0.0019, -0.0003, -0.0000,  -0.0001, -0.0527, -0.3894, -0.0527, -0.0001, -0.0000, -0.0000, -0.0000, -0.0000,  0.0000,  0.0001,  0.0527,  0.3894,  0.0527,  0.0001, 0.0000, 0.0003, 0.0019, 0.0003, 0.00};
   double Ay[] = {-0.0000, -0.0001, 0.0000, 0.0001, 0.0000, -0.0003, -0.0527,  0.0000,  0.0527, 0.0003, -0.0019, -0.3894,  0.0000,  0.3894, 0.0019, -0.0003, -0.0527, -0.0000, 0.0527, 0.0003, -0.0000, -0.0001, -0.0000, 0.0001, 0.0000};
   char *tran="N", retU = 'N', retV = 'A';
   ptrdiff_t lwork;
   int max1, max2;
   double *sss;
   
   /*printf("\n Recd %d features of size %d", ncols, mrows);*/
   
   /* Get input pointers */
   img = (double *)(mxGetPr(IMG_IN));
   wsz = (int)(mxGetScalar(WSZ_IN));
   h = (double)(mxGetScalar(H_IN));
   d = (int)(wsz*wsz); d2 = d;
   
   
   /* Input image is padded with (wsize-1)/2 + 2 in each side */
   mrows = mxGetM(IMG_IN); /*Row size - height of image*/
   ncols = mxGetN(IMG_IN); /* Column size - width of image*/
   
   /* The initial gradient window is 5x5 */
   gradHeight = mrows - 4;
   gradWidth = ncols - 4;
   height = mrows - wsz - 3;
   width = ncols - wsz - 3;
   
 /*  printf("\nImage size %dx%d",height,width);
   printf("\nGradient image %dx%d",gradHeight, gradWidth);*/
  /* printf(" to cluster into %d parts", K);
     
   printf("\n Calculating initial gradients....");*/
   
   yp = (double *)malloc(d2*sizeof(double));
   zx = (double *)malloc(gradHeight*gradWidth*sizeof(double));
   zy = (double *)malloc(gradHeight*gradWidth*sizeof(double));
   
   /* Get an initial estimation of the gradients with classical kernel regression 
   *  In the Matlab code, this part is implemented by ckr2reg.m */
   
   gw = gradWidth;
   gh = gradHeight;
   
   /* Fix the size of blocks for dgesvd */
   max1 = max(min(d,2)+3*2,2*min(d,2)+3);
   max2 = max(3*min(d,2)+max(d,2),5*min(d,2)-4);
   lwork = max(max1,max2);
   
   
   for(r=0; r<gradHeight; r++){
       for(c=0; c<gradWidth; c++){
           
           /* Form the patch centered at (r,c) */
           for(l=0, rptr=r*ncols; l<five; l++, rptr+=ncols)
                dcopy((mwSignedIndex*)&five, img + rptr + c, &inr, yp+l*five, &inr);
           
           /*printf("\n%3.3f, %3.3f, ", *(img+r*ncols + c + 3), *(yp+3));*/
           /* zx(r,c) = Ax*yp; zy(r,c) = Ay*yp; -- Get initial gradient */
           *(zx + r*gradWidth + c) = ddot(&twenty5, Ax, &inr, yp, &inr);
           *(zy + r*gradWidth + c) = ddot(&twenty5, Ay, &inr, yp, &inr);
            
           /*printf("%3.3f, %3.3f", *(zx+r*gradWidth+c), *(zy + r*gradWidth+c));     */
       }
   }
   
   free(yp);
  /* printf("Done\n Calculating Covariances....");*/
   
   /* Now obtain the steering kernels */
   G = (double *)malloc(2*d*sizeof(double));
   u = NULL; /*(double *)malloc(2*d2*sizeof(double));
   
   /* Dummy SVD run to get optimal blocks for Lapack */
   /*info = svdFunc(G, u, s, vt, d, 2);*/
   
  /* for(l=0, rptr = 0; l<width; l++,rptr+=gw){ 
         dcopy((mwSignedIndex*)&wsz, zx + rptr, &inr, G, &inr);
         dcopy((mwSignedIndex*)&wsz, zy + rptr, &inr, G+d, &inr);
   }
   
   lwork = -1;  work = (double*)mxCalloc(1,sizeof(double));  
   dgesvd(&retU, &retV, &d2, &two, G, &d2, s, u, &d2, vt, &two, work, &lwork, &info);
   mxFree(work);
   /*
   *work = (221.0*221.0/4.0);
   lwork = (ptrdiff_t) work[0];    
   
   work = (double*)mxRealloc(work, lwork*sizeof(double));*/
   
   
   work = (double *)mxCalloc(1, lwork*sizeof(double)); 
     
   /* Now form the local covariance matrices */
  /* C11 = (double *)malloc(gradHeight*gradWidth*sizeof(double)); 
   C12 = (double *)malloc(gradHeight*gradWidth*sizeof(double)); 
   C22 = (double *)malloc(gradHeight*gradWidth*sizeof(double)); */
   
   C11 = (double *)mxCalloc(gradHeight*gradWidth, sizeof(double)); 
   C12 = (double *)mxCalloc(gradHeight*gradWidth, sizeof(double)); 
   C22 = (double *)mxCalloc(gradHeight*gradWidth, sizeof(double));
   
   rad = (wsz-1)/2;
   /*vt = (double *)malloc(4*sizeof(double));
   s = (double *)malloc(2*sizeof(double));*/
   vt = (double *)mxCalloc(4,sizeof(double));
   s = (double *)mxCalloc(2, sizeof(double));
   
   for(r=0, r2=rad; r<height; r++, r2++){
       /*printf("\nStill working...");*/
       for(c=0, c2=rad; c<width; c++, c2++){
           
           /* Form the gradient matrix G for patch centered at (r,c) */
           for(l=0, rptr = r*gradWidth + c; l<wsz; l++, rptr+=gradWidth){ 
                dcopy((mwSignedIndex*)&wsz, zx + rptr, &inr, G+l*wsz, &inr);
                dcopy((mwSignedIndex*)&wsz, zy + rptr, &inr, G+d + l*wsz, &inr);
           }
           
           /* Get SVD of G */
          /* printf("\n%3.3f, %3.3f", *(zx+r*gradWidth+c), *(zy + r*gradWidth+c));  */
         /*printf("\n%3.3f, %3.3f: %3.3f, %3.3f", *(zx+r*gradWidth+c), *(zy + r*gradWidth+c), *G, *(G+d));  */
        dgesvd( &retU, &retV, &d2, &two, G, &d2, s, u, &d2, vt, &two, work, &lwork, &info );
          /* info = svdFunc(G, u, s, vt, d, 2);*/
         
        /* Check for convergence */
        if( info > 0 ) {
                printf( "The algorithm computing SVD failed to converge.\n" );
                exit( 1 );
        }
         
         /*tmp = (S(1) * v(:,1) * v(:,1).' + S(2) * v(:,2) * v(:,2).')  * (s(1,1) * s(2,2) + 0.0000001)^alpha;*/
        
         /*printf("\n %3.3f, %3.3f", s[0], s[1]);*/

         
         ss = sqrt((s[0] * s[1] + 0.0000001)/d2); 
         S[0] = (s[0] + One) / (s[1] + One);
         S[1] = 1.0/S[0];
         
         /**(sss + r*height + c) = -vt[0];*/
         
         S[0] = S[0]*ss; S[1] = S[1]*ss;
          
         
         /*printf("\n %3.3f, %3.3f",vt[0], vt[3]);*/
         /*dger(&two, &two, S, vt, &One, vt, &One, &tmp, &One); /* tmp = S(1)*v1*v1'; */
         /*dger(&two, &two, S+1, vt+2, &One, vt+2, &One, &tmp, &One); /* tmp += S(2)*v2*v2'; */
         
         /* return from Fortan dgesvd messes up the vt matrix */
         *(C11 + r2*gradWidth + c2) = S[0]*vt[2]*vt[2] + S[1]*vt[0]*vt[0];
         *(C12 + r2*gradWidth + c2) = S[0]*vt[2]*vt[3] + S[1]*vt[0]*vt[1];
         *(C22 + r2*gradWidth + c2) = S[0]*vt[3]*vt[3] + S[1]*vt[1]*vt[1];
           
       }
    }
   
   free(zx);
   free(zy);
   free(G);
   /*free(u);*/
   mxFree(vt);
   mxFree(s);
   mxFree(work); 
  /* printf("Done\n Padding covariance....");*/
  
    /* Pad C11, C12, C22 with symmetric borders */
   
   gw = width;
   for(r=0; r<rad; r++){
       
       r2 = (wsz - r - 1)*gradWidth + rad;
       dcopy(&gw, C11+r2, &inr, C11 + r*gradWidth + rad, &inr);
       dcopy(&gw, C12+r2, &inr, C12 + r*gradWidth + rad, &inr);
       dcopy(&gw, C22+r2, &inr, C22 + r*gradWidth + rad, &inr);
       
       /*r2 = (gradHeight - rad - r)*gradWidth + rad;*/
       r2 = (height + rad - 1 - r)*gradWidth + rad;
       dcopy(&gw, C11+r2, &inr, C11 + (r + height + rad)*gradWidth + rad, &inr);
       dcopy(&gw, C12+r2, &inr, C12 + (r + height + rad)*gradWidth + rad, &inr);
       dcopy(&gw, C22+r2, &inr, C22 + (r + height + rad)*gradWidth + rad, &inr);
       
   }
               
   gh = gradHeight;
   gw = gradWidth;
   for(c=0; c<rad; c++){
       
       c2 = (wsz - c - 1);
       dcopy(&gh, C11+c2, &gw, C11 + c, &gw);
       dcopy(&gh, C12+c2, &gw, C12 + c, &gw);
       dcopy(&gh, C22+c2, &gw, C22 + c, &gw);
       
       c2 = (width + rad - 1 - c);
      
       dcopy(&gh, C11+c2, &gw, C11 + c + width + rad, &gw);
       dcopy(&gh, C12+c2, &gw, C12 + c + width + rad, &gw);
       dcopy(&gh, C22+c2, &gw, C22 + c + width + rad, &gw);
      /* printf("\nCopying %d to %d", c2, c + width + rad);*/
   }
   
   
   
    /* Form (x - xi) */
   
   
   
   dx = (double *)malloc(d*sizeof(double));
   dy = (double *)malloc(d*sizeof(double));
           
   for(r=-rad, l=0; r<=rad; r++){
       for(c=-rad; c<=rad; c++){
            *(dy + l) = c;
            *(dx + l) = r;
            l++;
       }
   }
   
   W_OUT = mxCreateDoubleMatrix(1,height*width*d,mxREAL);
   W = (double *)(mxGetPr(W_OUT)); 
   
  /* printf("Done\n Forming LARK features....");*/
   
   
   /* The actual kernel computation */
   /* Remember, Matlab forms image patches column-wise */
   h2 = -0.5/(h*h);
   
   
   for(r=0, rptr=0; r<height; r++){
       for(c=0; c<width; c++){
           
           for(c2=0, l=0; c2<wsz; c2++){
               Cidx = r*gradWidth + c + c2;
               
               for(r2=0; r2<wsz; r2++, Cidx+=gradWidth){
                   *(W + rptr + l) = exp( h2*(*(C11+Cidx)*dx[l]*dx[l] + 2*(*(C12 + Cidx))*dx[l]*dy[l] + *(C22 + Cidx)*dy[l]*dy[l] ));
                   l++; 
                    
               }
           } 
           
           rptr += d;
       }
   }
   
  /* printf("Done\n");*/


   /*mxFree( (void*) work);*/
   
   mxFree(C11);
   mxFree(C12);
   mxFree(C22);
   free(dx);
   free(dy);
 }

