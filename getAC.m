function ac=getAC(ty,py)
% ty: true class
% py: predict class
if size(ty,1)~=size(py,1) || size(ty,2)~=size(py,2)
    error('The size of two vector is not the same in getAC.m');
end
m=size(ty,1);
tM=zeros(m*(m-1)/2,1);
pM=zeros(m*(m-1)/2,1);
num=1;
column=1;
j=1;
for i=1:m-1
    column=column+1;
    j=column;
    while j~=m+1 
        if ty(i,1)==ty(j,1)
            tM(num,1)=1;
        end
        if py(i,1)==py(j,1)
            pM(num,1)=1;
        end
        num=num+1;
        j=j+1;
    end
end
ac=size(find(tM==pM),1)/(m*(m-1)/2);
end