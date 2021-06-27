folder='F:\Shukai\NRL data\NRL - Connected Cavity Data\NRL - Connected Cavity Data\';
npick=200;

load([folder '1cavity_ports_on_same_face.mat']);
%load([folder '1cavity_ports_on_adjacent_faces.mat']);
for i=1:npick
    j=round(i*(200/npick));
    S11a(i,:)=reshape(SCf(:,1,j),16001,1);
    S12a(i,:)=reshape(SCf(:,2,j),16001,1);
    S21a(i,:)=reshape(SCf(:,3,j),16001,1);
    S22a(i,:)=reshape(SCf(:,4,j),16001,1);

end

load([folder '2cavities_port2_in_side_location.mat']);
%load([folder '2cavities_port2_in_top_location.mat']);

for i=1:npick
    j=round(i*(200/npick));
    S11b(i,:)=reshape(SCf(:,1,j),16001,1);
    S12b(i,:)=reshape(SCf(:,2,j),16001,1);
    S21b(i,:)=reshape(SCf(:,3,j),16001,1);
    S22b(i,:)=reshape(SCf(:,4,j),16001,1);

end

load([folder '3cavities_parrallel_port_locations.mat']);
%load([folder '3cavities_asymmetric_port_location.mat'']);
for i=1:npick
    j=round(i*(200/npick));
    S11c(i,:)=reshape(SCf(:,1,j),16001,1);
    S12c(i,:)=reshape(SCf(:,2,j),16001,1);
    S21c(i,:)=reshape(SCf(:,3,j),16001,1);
    S22c(i,:)=reshape(SCf(:,4,j),16001,1);

end
clear SCf
%% binary sort
n1_trn=npick*0.9;
n2_trn=n1_trn;
n_trn=n1_trn+n2_trn;
s1=0; s2=0;
while (s1~=n1_trn || s2~=n1_trn)
    for i=1:n_trn
       a=rand();
       if a<n1_trn/n_trn
           y_trn(i)=0;        % 0 is 1cav
       else
           y_trn(i)=1;        % 1 is 3cav
       end
    end
    [~,s1]=size(find(y_trn==0));
    [~,s2]=size(find(y_trn==1));
end
size(find(y_trn==0))  %check random
size(find(y_trn==1))  %check random
%% arrange test
n1_tst=npick*0.1;
n2_tst=n1_tst;
n_tst=n1_tst+n2_tst;
s1=0; s2=0;
while (s1~=n1_tst || s2~=n1_tst)
    for i=1:n_tst
       a=rand();
       if a<n1_tst/n_tst
           y_tst(i)=0;        % 0 is 1cav
       else
           y_tst(i)=1;        % 1 is 3cav
       end
    end
    [~,s1]=size(find(y_tst==0));
    [~,s2]=size(find(y_tst==1));
end
size(find(y_tst==0))  %check random
size(find(y_tst==1))  %check random
%%
npick=10000;
%% method1, front pick
clear x_tst x_trn 
ia=1; ib=ia;
for i=1:n_tst
    if y_tst(i)>0
        x_tst(:,i)=S11a(ib,1:npick);
        ib=ib+1;
    else
        x_tst(:,i)=S11c(ia,1:npick);
        ia=ia+1;
    end
end

for i=1:n_trn
    if y_trn(i)>0
        x_trn(:,i)=S11a(ib,1:npick);
        ib=ib+1;
    else
        x_trn(:,i)=S11c(ia,1:npick);
        ia=ia+1;
    end
end
%% pick messed up data
ntrick=5;
for i=1:ntrick
    a=rand();
    np=floor(a*n_tst);
    list(1,i)=np;
end
ia=1;
for i=1:ntrick
    y_tst(list(i))=2; % 2 means 2cav 
    x_tst(:,list(i))=S11b(ia,1:npick);
    ia=ia+1;
end
%% total mess
for i=1:n_tst
    y_tst(i)=2; % 2 means 2cav 
    x_tst(:,i)=S11b(i,1:npick);
end

%% h5 write, train
[a,b]=size(x_trn);
h5out=['train_trick11a.h5'];
h5create(h5out,'/train_set_x',[a b]);
h5write(h5out,'/train_set_x',x_trn);

[a,b]=size(y_trn);
h5create(h5out,'/train_set_y',[max(a,b)],'Datatype','int32');
h5write(h5out,'/train_set_y',y_trn);
%
%for i=1:2
    
%   class(i)=i-1; 
%end

for i=1:2
    
   class(i)=i-1; 
end

[a,b]=size(class);
h5create(h5out,'/list_classes',[b],'Datatype','int32');
h5write(h5out,'/list_classes',class);

% h5 write, test
[a,b]=size(x_tst);
h5out=['test_trick11a.h5'];
h5create(h5out,'/test_set_x',[a b]);
h5write(h5out,'/test_set_x',x_tst);

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