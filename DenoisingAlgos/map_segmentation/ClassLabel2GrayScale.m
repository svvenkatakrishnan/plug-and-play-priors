function [Imout,ClassMean,ClassVar]=ClassLabel2GrayScale(Img,Label,NumClass)
%Converts a segmentation Label field into an "image" by finding the mean
%value of the original image in different regions using the label map. The
%Inputs: Img: Original image used for segmentation
%        Label : A image of same size as Img with pixels representing class
%        labels
%        NumClass: Number of classes in the image (same as max(Label(:))
%Outputs: Imout: A new image with pixel values representing class means
%         ClassMean : Mean value of each class
%         ClassVar  : Variance of each class
Imout=zeros(size(Img));
Img=double(Img);
ClassMean=zeros(1,NumClass);
ClassVar = zeros(1,NumClass);
for i=1:NumClass
    ClassMean(i)=mean(Img(Label==i));
    ClassVar(i)= var(Img(Label==i));
    Imout(Label==i)=ClassMean(i);
end