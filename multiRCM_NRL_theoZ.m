% make new test sets for the theory generated Zs
folder='C:\Users\skma\PycharmProjects\RCMclsfy_nrl\datasets\';
load([folder 'theoZin_2and3.mat']);
%%
npick=200;

folder='F:\Shukai\NRL data\higherloss\Connected Cavity Sparam Data - Higher Alpha\';
load([folder '1p6a.mat']);
%load([folder '2cavities_port2_in_top_location.mat']);

for i=1:npick
    j=round(i*(200/npick));
    S11a(i,:)=reshape(SCf(:,1,j),16001,1);
    S12a(i,:)=reshape(SCf(:,2,j),16001,1);
    S21a(i,:)=reshape(SCf(:,3,j),16001,1);
    S22a(i,:)=reshape(SCf(:,4,j),16001,1);

end
[n,numpoints]=size(S11a);
freq_start=3.95*10^9;
freq_end=5.85*10^9;
freq=linspace(freq_start, freq_end, numpoints);
Z0=50;
[Z11a,Z12a,Z21a,Z22a]=S2Z_Bo(S11a,S12a,S21a,S22a,Z0,Z0); % 50 is the Z for transmission line
%% arrange test
clear y_tst
numt=20;
n1_tst=numt;
n2_tst=numt;
n3_tst=numt;
n_tst=n1_tst+n2_tst+n3_tst;
s1=0; s2=0;
while (s1~=numt || s2~=numt)
    for i=1:n_tst
       a=rand();
       if a<n1_tst/n_tst
           y_tst(i)=0;        % 0 is 1cav
       elseif a<(n1_tst+n2_tst)/n_tst
           y_tst(i)=1;        % 1 is 2cav
       else
           y_tst(i)=2;
       end
    end
    [~,s1]=size(find(y_tst==0));
    [~,s2]=size(find(y_tst==2));
end
size(find(y_tst==0))  %check random
size(find(y_tst==1))  %check random
size(find(y_tst==2))  %check random

%%
npick=1000;
ntot=16001;
Z11b=Ztheo_in2_1000;
Z11c=Ztheo_in3_1000;
clear x_tst x_trn ind
ia=1; ib=ia; ic=ia;

for i=1:npick
   ind(i)=floor(ntot/npick)*(i-1)+1; 
end


for i=1:n_tst
    if y_tst(i)==0
        x_tst(:,i)=Z11a(ia,ind);
        ia=ia+1;
    elseif y_tst(i)==1
        x_tst(:,i)=Z11b(ib,:);
        ib=ib+1;
    else
        x_tst(:,i)=Z11c(ic,:);
        ic=ic+1;
    end
end

%% h5 write, test
[a,b]=size(x_tst);
h5out=['test_rcm_nrl_3_Z11_mix.h5'];
h5create(h5out,'/test_set_x',[a b]);
h5write(h5out,'/test_set_x',real(x_tst));

[a,b]=size(y_tst);
h5create(h5out,'/test_set_y',[max(a,b)],'Datatype','int32');
h5write(h5out,'/test_set_y',y_tst);
%
for i=1:2
   class(i)=i-1; 
end
[a,b]=size(class);
h5create(h5out,'/list_classes',[b],'Datatype','int32');
h5write(h5out,'/list_classes',class);