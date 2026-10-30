function evaluate_mmt(input_file, output_file)
addpath(fullfile(fileparts(mfilename('fullpath')), 'mmt_utils'));
data = load(input_file);
y0 = data.y0;
lam = data.lam;
nY = size(y0,2);
Y = zeros(nY,1);
if nY < 8
    for i = 1:nY
        temp = mmt_nonlinear_stoch_y0(y0(:,i),1,lam);
        Y(i) = max(abs(real(temp(end,:))));
    end
else
    parfor i = 1:nY
        temp = mmt_nonlinear_stoch_y0(y0(:,i),1,lam);
        Y(i) = max(abs(real(temp(end,:))));
    end
end
save(output_file, 'Y');
end
