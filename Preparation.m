files = dir(fullfile(pwd));
dirFlags = [files.isdir];
Folders = files(dirFlags);
Folders = {Folders.name};
l = 1;
a = 512;
b = 512;

D0 = 150;
A = 2*a + 1;
B = 2*b + 1;
P = ceil(B/2);
Q = ceil(A/2);
[u, v] = meshgrid(1:B,1:A);
D = (u-P).^2 + (v-Q).^2;
H = exp(-D/(2*(D0^2)));
H = fftshift(H);

for i = 3:length(Folders)
    files = dir(fullfile(pwd, Folders{1,i}));
    sys = {files.name};
    for m = 3:length(sys)
        tmp = imread([Folders{1,i}, '\', sys{m}]);
        if size(tmp,3) == 3
            tmp = rgb2gray(tmp);
        end
        c = floor(size(tmp,1)/a);
        d = floor(size(tmp,2)/b);

        if c > 0 && d > 0
            for j = 1:c
                for k = 1:d
                    l
                    y(:,:,l) = tmp((j-1)*a+1:j*a, (k-1)*b+1:k*b);
                    tmp1 = glpf(y(:,:,l), H, A, B);
                    x(:,:,l) = tmp1(1:2:end,1:2:end);
                    l = l+1;
                end
            end
        end
    end
end

p = randperm(size(x,3));
x = permute(x(:,:,p), [3,1,2]);
y = permute(y(:,:,p), [3,1,2]);

xtest = x(1:1000,:,:);
ytest = y(1:1000,:,:);

x = x(1001:end,:,:);
y = y(1001:end,:,:);

save('imagedata.mat','x','y','-v7.3');
save('imagetestdata.mat','xtest','ytest');