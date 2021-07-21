load('Prediction.mat');
nn1 = permute(ytesthat, [2,3,1]);
load('imagetestdata.mat');
small = permute(xtest, [2,3,1]);
truth = permute(ytest, [2,3,1]);
load('bicubictest.mat');
bicubic = permute(xtest, [2,3,1]);
load('BicubicPrediction.mat');
nn2 = permute(ytesthat, [2,3,1]);

for i = 1:size(nn1,3)
    I{i,1} = small(:,:,i);
    I{i,2} = bicubic(:,:,i);
    I{i,3} = nn1(:,:,i);
    I{i,4} = nn2(:,:,i);
    I{i,5} = truth(:,:,i);
end

save('TestResults.mat','I','-v7.3');