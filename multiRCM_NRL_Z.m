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
%% S to Z, set Z0=50
[n,numpoints]=size(S11a);
freq_start=3.95*10^9;
freq_end=5.85*10^9;
freq=linspace(freq_start, freq_end, numpoints);
Z0=50;
[Z11a,Z12a,Z21a,Z22a]=S2Z_Bo(S11a,S12a,S21a,S22a,Z0,Z0); % 50 is the Z for transmission line
[Z11b,Z12b,Z21b,Z22b]=S2Z_Bo(S11b,S12b,S21b,S22b,Z0,Z0); % 50 is the Z for transmission line
[Z11c,Z12c,Z21c,Z22c]=S2Z_Bo(S11c,S12c,S21c,S22c,Z0,Z0); % 50 is the Z for transmission line
%%
mk=7;
fig1=plot(freq/10^9,imag(Z11a(mk,:)),'-','DisplayName','1cav'); hold on;
fig1=plot(freq/10^9,imag(Z11b(mk,:)),'-','DisplayName','2cav'); hold on;
fig1=plot(freq/10^9,imag(Z11c(mk,:)),'-','DisplayName','3cav'); hold on;
title('Z_{11} imag part compare');
xlabel('Frequency (GHz)');
ylabel('Ime(Z) (\Omega)');
box on;
xlim([3.95 5.85]);
legend;
hold off
%% binary sort
n1_trn=180;
n2_trn=180;
n3_trn=180;
n_trn=n1_trn+n2_trn+n3_trn;
s1=0; s2=0;
while (s1~=180 || s2~=180)
    for i=1:n_trn
       a=rand();
       if a<n1_trn/n_trn
           y_trn(i)=0;        % 0 is 1cav
       elseif a<(n1_trn+n2_trn)/n_trn
           y_trn(i)=1;        % 1 is 2cav
       else
           y_trn(i)=2;
       end
    end
    [~,s1]=size(find(y_trn==0));
    [~,s2]=size(find(y_trn==2));
end
size(find(y_trn==0))  %check random
size(find(y_trn==2))  %check random
%% arrange test
n1_tst=20;
n2_tst=20;
n3_tst=20;
n_tst=n1_tst+n2_tst+n3_tst;
s1=0; s2=0;
while (s1~=20 || s2~=20)
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
size(find(y_tst==2))  %check random
%%
npick=10000;
%% method1, front pick
clear x_tst x_trn 
ia=1; ib=ia; ic=ia;
for i=1:n_tst
    if y_tst(i)==0
        x_tst(:,i)=Z11a(ia,1:npick);
        ia=ia+1;
    elseif y_tst(i)==1
        x_tst(:,i)=Z11b(ib,1:npick);
        ib=ib+1;
    else
        x_tst(:,i)=Z11c(ic,1:npick);
        ic=ic+1;
    end
end

for i=1:n_trn
    if y_trn(i)==0
        x_trn(:,i)=Z11a(ia,1:npick);
        ia=ia+1;
    elseif y_trn(i)==1
        x_trn(:,i)=Z11b(ib,1:npick);
        ib=ib+1;
    else
        x_trn(:,i)=Z11c(ic,1:npick);
        ic=ic+1;
    end
end
%%
%x_tst=x_tst-mean(x_tst)./std(x_tst);
%x_trn=x_trn-mean(x_trn)./std(x_trn);
%% h5 write, train
[a,b]=size(x_trn);
h5out=['train_rcm_nrl_3_Z11.h5'];
h5create(h5out,'/train_set_x',[a b]);
h5write(h5out,'/train_set_x',real(x_trn));

[a,b]=size(y_trn);
h5create(h5out,'/train_set_y',[max(a,b)],'Datatype','int32');
h5write(h5out,'/train_set_y',y_trn);
%
for i=1:2
   class(i)=i-1; 
end
[a,b]=size(class);
h5create(h5out,'/list_classes',[b],'Datatype','int32');
h5write(h5out,'/list_classes',class);
% h5 write, test
[a,b]=size(x_tst);
h5out=['test_rcm_nrl_3_Z11.h5'];
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
%% real+imag
clear x_tstt x_trnt 
ia=1; ib=ia; ic=ia;
for i=1:n_tst
    if y_tst(i)==0
        x_tst11(:,i)=Z11a(ia,1:npick);
        ia=ia+1;
    elseif y_tst(i)==1
        x_tst11(:,i)=Z11b(ib,1:npick);
        ib=ib+1;
    else
        x_tst11(:,i)=Z11c(ic,1:npick);
        ic=ic+1;
    end
end

for i=1:n_trn
    if y_trn(i)==0
        x_trn11(:,i)=Z11a(ia,1:npick);
        ia=ia+1;
    elseif y_trn(i)==1
        x_trn11(:,i)=Z11b(ib,1:npick);
        ib=ib+1;
    else
        x_trn11(:,i)=Z11c(ic,1:npick);
        ic=ic+1;
    end
end


[a,b]=size(x_tst11);
x_tstt=zeros(2*a,b);
x_tstt(1:a,:)=real(x_tst11);
x_tstt(a+1:2*a,:)=imag(x_tst11);


[a,b]=size(x_trn11);
x_trnt=zeros(2*a,b);
x_trnt(1:a,:)=real(x_trn11);
x_trnt(a+1:2*a,:)=imag(x_trn11);

%% h5 write, train
[a,b]=size(x_trnt);
h5out=['train_rcm_nrl_Zall.h5'];
h5create(h5out,'/train_set_x',[a b]);
h5write(h5out,'/train_set_x',(x_trnt));

[a,b]=size(y_trn);
h5create(h5out,'/train_set_y',[max(a,b)],'Datatype','int32');
h5write(h5out,'/train_set_y',(y_trn));
%
for i=1:2
    
   class(i)=i-1; 
end
[a,b]=size(class);
h5create(h5out,'/list_classes',[b],'Datatype','int32');
h5write(h5out,'/list_classes',class);
% h5 write, test
[a,b]=size(x_tstt);
h5out=['test_rcm_nrl_Zall.h5'];
h5create(h5out,'/test_set_x',[a b]);
h5write(h5out,'/test_set_x',x_tstt);

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


%% all-in
clear x_tstt x_trnt 
ia=1; ib=ia; ic=ia;
for i=1:n_tst
    if y_tst(i)==0
        x_tst11(:,i)=Z11a(ia,1:npick);
        ia=ia+1;
    elseif y_tst(i)==1
        x_tst11(:,i)=Z11b(ib,1:npick);
        ib=ib+1;
    else
        x_tst11(:,i)=Z11c(ic,1:npick);
        ic=ic+1;
    end
end

for i=1:n_trn
    if y_trn(i)==0
        x_trn11(:,i)=Z11a(ia,1:npick);
        ia=ia+1;
    elseif y_trn(i)==1
        x_trn11(:,i)=Z11b(ib,1:npick);
        ib=ib+1;
    else
        x_trn11(:,i)=Z11c(ic,1:npick);
        ic=ic+1;
    end
end

ia=1; ib=ia; ic=ia;
for i=1:n_tst
    if y_tst(i)==0
        x_tst12(:,i)=Z12a(ia,1:npick);
        ia=ia+1;
    elseif y_tst(i)==1
        x_tst12(:,i)=Z12b(ib,1:npick);
        ib=ib+1;
    else
        x_tst12(:,i)=Z12c(ic,1:npick);
        ic=ic+1;
    end
end

for i=1:n_trn
    if y_trn(i)==0
        x_trn12(:,i)=Z12a(ia,1:npick);
        ia=ia+1;
    elseif y_trn(i)==1
        x_trn12(:,i)=Z12b(ib,1:npick);
        ib=ib+1;
    else
        x_trn12(:,i)=Z12c(ic,1:npick);
        ic=ic+1;
    end
end

ia=1; ib=ia; ic=ia;
for i=1:n_tst
    if y_tst(i)==0
        x_tst21(:,i)=Z21a(ia,1:npick);
        ia=ia+1;
    elseif y_tst(i)==1
        x_tst21(:,i)=Z21b(ib,1:npick);
        ib=ib+1;
    else
        x_tst21(:,i)=Z21c(ic,1:npick);
        ic=ic+1;
    end
end

for i=1:n_trn
    if y_trn(i)==0
        x_trn21(:,i)=Z21a(ia,1:npick);
        ia=ia+1;
    elseif y_trn(i)==1
        x_trn21(:,i)=Z21b(ib,1:npick);
        ib=ib+1;
    else
        x_trn21(:,i)=Z21c(ic,1:npick);
        ic=ic+1;
    end
end

ia=1; ib=ia; ic=ia;
for i=1:n_tst
    if y_tst(i)==0
        x_tst22(:,i)=Z22a(ia,1:npick);
        ia=ia+1;
    elseif y_tst(i)==1
        x_tst22(:,i)=Z22b(ib,1:npick);
        ib=ib+1;
    else
        x_tst22(:,i)=Z22c(ic,1:npick);
        ic=ic+1;
    end
end

for i=1:n_trn
    if y_trn(i)==0
        x_trn22(:,i)=Z22a(ia,1:npick);
        ia=ia+1;
    elseif y_trn(i)==1
        x_trn22(:,i)=Z22b(ib,1:npick);
        ib=ib+1;
    else
        x_trn22(:,i)=Z22c(ic,1:npick);
        ic=ic+1;
    end
end
[a,b]=size(x_tst11);
x_tstt=zeros(2*a*4,b);
x_tstt(1:a,:)=real(x_tst11);
x_tstt(a+1:2*a,:)=imag(x_tst11);
x_tstt(2*a+1:3*a,:)=real(x_tst12);
x_tstt(3*a+1:4*a,:)=imag(x_tst12);
x_tstt(4*a+1:5*a,:)=real(x_tst21);
x_tstt(5*a+1:6*a,:)=imag(x_tst21);
x_tstt(6*a+1:7*a,:)=real(x_tst22);
x_tstt(7*a+1:8*a,:)=imag(x_tst22);

[a,b]=size(x_trn11);
x_trnt=zeros(2*a*4,b);
x_trnt(1:a,:)=real(x_trn11);
x_trnt(a+1:2*a,:)=imag(x_trn11);
x_trnt(2*a+1:3*a,:)=real(x_trn12);
x_trnt(3*a+1:4*a,:)=imag(x_trn12);
x_trnt(4*a+1:5*a,:)=real(x_trn21);
x_trnt(5*a+1:6*a,:)=imag(x_trn21);
x_trnt(6*a+1:7*a,:)=real(x_trn22);
x_trnt(7*a+1:8*a,:)=imag(x_trn22);
%% h5 write, train
[a,b]=size(x_trnt);
h5out=['train_rcm_nrl_3al.h5'];
h5create(h5out,'/train_set_x',[a b]);
h5write(h5out,'/train_set_x',(x_trnt));

[a,b]=size(y_trn);
h5create(h5out,'/train_set_y',[max(a,b)],'Datatype','int32');
h5write(h5out,'/train_set_y',(y_trn));
%
for i=1:2
    
   class(i)=i-1; 
end
[a,b]=size(class);
h5create(h5out,'/list_classes',[b],'Datatype','int32');
h5write(h5out,'/list_classes',class);
% h5 write, test
[a,b]=size(x_tstt);
h5out=['test_rcm_nrl_3al.h5'];
h5create(h5out,'/test_set_x',[a b]);
h5write(h5out,'/test_set_x',x_tstt);

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