%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           HS_splice performs a logistic splicing correction on a pair of co-registered 
%           hyperspectral images that present overlapping wavelengths intervals.
% 
% Usage:
%           [full_cube,wvl_full] = HS_splice(vnir_cube,wvl_vnir,swir_cube,wvl_swir,move_factor,mask)
%
% Inputs:
%           vnir_cube : VNIR hyperspectral image in double data type and size MxNxV
%           wvl_vnir : vector of size Vx1 of wavelengths in the VNIR spectral sampling    
%           swir_cube : SWIR hyperspectral image in double data type and size MxNxS
%           wvl_swir : vector of size Sx1 of wavelengths in the SWIR spectral sampling  
%           move_factor : factor that quantifies how much the spectral curves move towards each other. Must be in the range [0,1]
%               0: SWIR curve used as reference
%               1: VNIR curve used as reference
%               .5: Match at mean between VNIR and SWIR curves
%           mask : binary map for segmented images, if not provided the
%               whole image is processed
%           
% Outputs:
%           full_cube: Spliced hyperspectral image of size MxNxF
%           wvl_full: vector of size Fx1 of wavelengths in the VNIR-SWIR spectral sampling
%
% MATLAB REQUIREMENTS:
%           This script was written on version 2021b
%           Suggested toolbox: STATISTICS AND MACHINE LEARNING, CURVE FITTING
% 
% References:
%           Please cite the following article if you ever use the contents of this script:
%           "F.Grillini, J.B. Thomas,  & S.George. (2022). A novel splicing correction for VNIR-SWIR imaging spectroscopy"
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [full_cube,wvl_full] = HS_splice(vnir_cube,wvl_vnir,swir_cube,wvl_swir,move_factor,mask)

    if nargin < 6
        mask = ones(size(vnir_cube,[1,2]),'logical');
    end
    check_arguments(vnir_cube,wvl_vnir,swir_cube,wvl_swir,move_factor,mask);

    % Overlap index in SWIR range
    wvl_ip_vnir_at_swir = interp1(wvl_vnir,wvl_vnir,wvl_swir);
    idx_overlap_at_swir = ~isnan(wvl_ip_vnir_at_swir);
    % Overlap wavelengths
    wvl_overlap = wvl_swir(idx_overlap_at_swir);
    % Interpolated VNIR at the overlap wavelengths
    vnir_cube_overlap = interp_vnir_overlap(vnir_cube,wvl_vnir,wvl_swir,idx_overlap_at_swir);
    % Index of bands outside of the overlap range
    idx_outside_at_swir = ~idx_overlap_at_swir;
    wvl_ip_swir_at_vnir = interp1(wvl_swir,wvl_swir,wvl_vnir);
    idx_outside_at_vnir = isnan(wvl_ip_swir_at_vnir);
    % Wavelengths outside of the overlap range
    outside_wvl_vnir = wvl_vnir(idx_outside_at_vnir);
    outside_wvl_swir = wvl_swir(idx_outside_at_swir);

    % Spectral discrepancy over the image
    Delta_R = sqrt(mean((vnir_cube_overlap - swir_cube(:,:,idx_overlap_at_swir)).^2,3));
    % Correcting coefficients in the overlap
    [coeff_overlap_vnir,coeff_overlap_swir] = overlap_coefficients(Delta_R,move_factor,...
        vnir_cube_overlap,swir_cube(:,:,idx_overlap_at_swir));

    % Extreme coefficients phi_v and phi_s
    phi_vnir = coeff_overlap_vnir(:,:,1);
    phi_swir = coeff_overlap_swir(:,:,end);

    % Fits of window size and slope (VNIR only)
    [fit_window_vnir,fit_window_swir,fit_slope_vnir] = find_fits(Delta_R,idx_outside_at_vnir,idx_outside_at_swir,mask); 
    % Vector of final wavelengths
    wvl_full = [outside_wvl_vnir;wvl_overlap;outside_wvl_swir];
    

    % Number of bands affected by the correction
    window_vnir = reshape(round(fit_window_vnir(Delta_R)),size(Delta_R));
    window_swir = reshape(round(fit_window_swir(Delta_R)),size(Delta_R));


    % Central wavelength affected by the correction
    lambda_0_vnir = ones(size(window_vnir))*median(outside_wvl_vnir);
    lambda_0_vnir(mask) = outside_wvl_vnir(end-round(window_vnir(mask)/2));
    lambda_0_swir = ones(size(window_swir))*median(outside_wvl_swir);
    lambda_0_swir(mask) = outside_wvl_swir(round(window_swir(mask)/2));

    % Model the slope of the logistic function in VNIR
    slope_vnir = zeros(size(window_vnir));
    slope_vnir(mask) = fit_slope_vnir(Delta_R(mask));

    % To model the slope of the logistic function in SWIR, use prior
    % information about the total number of bands outside of the
    % overlapping region
    slope_swir = slope_vnir.*(length(outside_wvl_vnir)/length(outside_wvl_swir));
    % Implementation of logistic function to converge coefficients to 1
    % Sign function 
    L = ones(size(Delta_R));
    L(phi_vnir<1) = -1;
    % Logistic function
    coeff_vnir = L./(1+exp(-slope_vnir.*(reshape(outside_wvl_vnir,[1,1,numel(outside_wvl_vnir)])...
        -lambda_0_vnir))).*(abs(phi_vnir-1))+1; 
    coeff_swir = L./(1+exp(-slope_swir.*(reshape(outside_wvl_swir,[1,1,numel(outside_wvl_swir)])...
        -lambda_0_swir))).*(abs(phi_swir-1))+phi_swir;

    full_cube = cat(3,vnir_cube(:,:,idx_outside_at_vnir).*coeff_vnir,...
       swir_cube(:,:,idx_overlap_at_swir).*coeff_overlap_swir,...
        swir_cube(:,:,idx_outside_at_swir).*coeff_swir);
end

function check_arguments(vnir_cube,wvl_vnir,swir_cube,wvl_swir,move_factor,mask)
    if ~isa(vnir_cube,'double')
        error('The VNIR image must be of double data type.')
    end

    if ~isa(swir_cube,'double')
        error('The SWIR image must be of double data type.')
    end

    if ~isequal(size(vnir_cube,3),numel(wvl_vnir))
        error('The number of bands of the VNIR image must equal the number of elements in the VNIR wavelength vector.')
    end

    if ~isequal(size(swir_cube,3),numel(wvl_swir))
        error('The number of bands of the SWIR image must equal the number of elements in the SWIR wavelength vector.')
    end

    if ~isequal(size(vnir_cube,[1,2]),size(swir_cube,[1,2]))
        error('The spatial size of VNIR and SWIR images must match.')
    end

    if max(wvl_vnir) < min(wvl_swir)
        error('The max value of VNIR wavelengths is smaller than the min value of SWIR wavelengths. No overlapping region exists.')
    end

    if ~isscalar(move_factor)
        error('The move factor must be a scalar in the interval [0,1].')
    end

    if (move_factor<0 || move_factor > 1)
        error('The move factor must be a scalar in the interval [0,1].')
    end
    
    if ~isequal(size(vnir_cube,[1,2]),size(mask))
        error('The spatial size of the images and the binary mask must match.')
    end

    
end


function ip_overlap_vnir = interp_vnir_overlap(vnir_cube,wvl_vnir,wvl_swir,idx)
    N = numel(find(idx));
    [row,col,bands] = size(vnir_cube);
    resh_v = reshape(vnir_cube,[row*col,bands])';
    ip_overlap_vnir = reshape(interp1(wvl_vnir,resh_v,wvl_swir(idx))',[row,col,N]);
end

function [fit_window_vnir,fit_window_swir,fit_slope_vnir] = find_fits(Delta_R,idx_out_vnir,idx_out_swir,mask)
    % Get discrepancy values to fit window size and slope
    Mdelta = prctile(Delta_R(mask),75)+1.5*((prctile(Delta_R(mask),75)-prctile(Delta_R(mask),25)));
    mdelta = prctile(Delta_R(mask),1);
    % Window fitting distribution
    leftout_bands_vnir = numel(find(idx_out_vnir));
    leftout_bands_swir = numel(find(idx_out_swir));
    if mdelta == 0
        mdelta = Mdelta/leftout_bands_vnir;
    end
    window_delta_vnir = logspace(log10(mdelta),log10(Mdelta),leftout_bands_vnir);
    window_delta_swir = logspace(log10(mdelta),log10(Mdelta),leftout_bands_swir);
    fitfun_w = fittype( 'a1/(1+exp(-b1*(x-c1)))', 'independent', 'x', 'dependent', 'y' );
    opts_wv = fitoptions( 'Method', 'NonlinearLeastSquares');
    opts_ws = fitoptions( 'Method', 'NonlinearLeastSquares');
    opts_wv.Display = 'Off'; opts_ws.Display = 'Off';
    opts_wv.StartPoint = [leftout_bands_vnir leftout_bands_vnir/2 median(window_delta_vnir)];
    opts_ws.StartPoint = [leftout_bands_swir leftout_bands_swir/2 median(window_delta_swir)];
    opts_wv.Upper = [leftout_bands_vnir inf 1]; 
    opts_ws.Upper = [leftout_bands_swir inf 1];
    fit_window_vnir = fit(window_delta_vnir.',(1:leftout_bands_vnir).',fitfun_w,opts_wv);
    fit_window_swir = fit(window_delta_swir.',(1:leftout_bands_swir).',fitfun_w,opts_ws);
    diff_fit3 = diff(diff(diff(fit_window_vnir(window_delta_vnir)))); %MAX of 3rd derivative
    [~,idx] = max(diff_fit3);
    Mdelta2 = window_delta_vnir(idx);
    window_delta_vnir2 = logspace(log10(mdelta),log10(Mdelta2),leftout_bands_vnir);
    fitfun_kv = fittype('a*exp(b*x)+c','independent','x','dependent','y');
    opts_kv =  fitoptions( 'Method', 'NonlinearLeastSquares');
    opts_kv.Display = 'Off'; 
    opts_kv.StartPoint = [0 0 0];
    opts_kv.Lower = [-inf -inf 0];
    opts_kv.Upper = [inf 0 1]; 
    norm_vector = ((leftout_bands_vnir:-1:1) - 1)/(leftout_bands_vnir-1);
    fit_slope_vnir = fit(window_delta_vnir2.',norm_vector.',fitfun_kv,opts_kv);
end

function [coeff_vnir,coeff_swir] = overlap_coefficients(Delta_R,move_factor,overlap_vnir,overlap_swir)
    difference = sum(overlap_vnir-overlap_swir,3);
    mask = repmat((difference>0),[1,1,size(overlap_swir,3)]);
    swir_move = Delta_R*move_factor; 
    vnir_move = Delta_R*(1-move_factor);
    R = zeros(size(mask,1),size(mask,2),size(overlap_swir,3));
    R_pos = ((overlap_vnir-vnir_move)+(overlap_swir+swir_move))/2;
    R_neg = ((overlap_vnir+vnir_move)+(overlap_swir-swir_move))/2;
    R(mask) = R_pos(mask);
    R(~mask) = R_neg(~mask);
    coeff_vnir = R ./ overlap_vnir; 
    coeff_swir = R ./ overlap_swir; 
end


