load('Prediction.mat');
load('imagetestdata.mat');

ytesthat = uint8(ytesthat);
xtest = permute(xtest, [2,3,1]);
ytest = permute(ytest, [2,3,1]);
ytesthat = permute(ytesthat, [2,3,1]);

bicubemse = 0;
nnmse = 0;

for i = 1:size(xtest,3)
    ybicubic(:,:,i) = imresize(xtest(:,:,i), 2, 'bicubic');
    bicubemse = bicubemse + immse(ybicubic(:,:,i), ytest(:,:,i));
    nnmse = nnmse + immse(ytesthat(:,:,i), ytest(:,:,i));
end

bicubemse = bicubemse/size(xtest,3);
nnmse = nnmse/size(xtest,3);