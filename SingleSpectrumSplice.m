%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%          SingleSpectrumSplice performs a logistic splicing correction on
%          a pair of spectra that share a certain overlap in their wavelength sampling
% 
% Usage:
%           [full_spectrum,wvl_full] = SingleSpectrumSplice(vnir_spectrum,wvl_vnir,swir_spectrum,wvl_swir,...
%               fit_window_vnir,fit_window_swir,fit_k_vnir,overlap_matching)
%
% Inputs:
%           vnir_spectrum : VNIR spectrum in double data type and size Vx1
%           wvl_vnir : vector of size Mx1 of wavelengths in the VNIR spectral sampling    
%           swir_spectrum : SWIR spectrum in double data type and size Sx1
%           wvl_swir : vector of size Sx1 of wavelengths in the SWIR spectral sampling  
%           fit_window_vnir : fitting function of the logistic function window in the VNIR spectral sampling
%           fit_window_swir : fitting function of the logistic function window in the SWIR spectral sampling
%           fit_k_vnir : fitting function of the logistic function slope in the VNIR spectral sampling
%           move_factor : factor that quantifies how much the spectral curves move towards each other. Must be in the range [0,1]
%               0: SWIR curve used as reference
%               1: VNIR curve used as reference
%               .5: Match at mean between VNIR and SWIR curves
%           mask : binary map for segmented images, if not provided the
%               whole image is processed
%
% MATLAB REQUIREMENTS:
%           This script was written on version 2021b
%           Suggested toolbox: STATISTICS AND MACHINE LEARNING, CURVE FITTING
%           
% Outputs:
%           full_spectrum : Spliced spectrum of size Fx1
%           wvl_full: vector of size Fx1 of wavelengths in the VNIR-SWIR spectral sampling
% 
% References:
%           Please cite the following article if you ever use the contents of this script:
%           "F.Grillini, J.B. Thomas,  & S.George. (2022). A novel splicing correction for VNIR-SWIR imaging spectroscopy"
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [full_spectrum,wvl_full] = SingleSpectrumSplice(vnir_spectrum,wvl_vnir,swir_spectrum,wvl_swir,...
    fit_window_vnir,fit_window_swir,fit_k_vnir,move_factor)
    
    % interpolate VNIR spectrum to match the sampling of SWIR
    ip_vnir_to_swir = interp1(wvl_vnir,vnir_spectrum,wvl_swir);
    % obtain VNIR and SWIR spectra in the overlap
    idx_overlap_at_swir = ~isnan(ip_vnir_to_swir);
    wvl_overlap = wvl_swir(idx_overlap_at_swir);
    overlap_vnir = ip_vnir_to_swir(idx_overlap_at_swir);
    overlap_swir = swir_spectrum(idx_overlap_at_swir);
    % Get the spectral discrepancy
    delta_r = sqrt(mean((overlap_vnir - overlap_swir).^2));
    % Get the correcting coefficients in the overlap region
    if delta_r == 0
        coeff_overlap_vnir = ones(length(wvl_overlap),1); 
        coeff_overlap_swir = ones(length(wvl_overlap),1); 
    else
        swir_move = delta_r*move_factor; 
        vnir_move = delta_r*(1-move_factor);
        if sum(overlap_vnir-overlap_swir) > 0
            R = mean([overlap_vnir-vnir_move,overlap_swir+swir_move],2);
        else
            R = mean([overlap_vnir+vnir_move,overlap_swir-swir_move],2);
        end
        coeff_overlap_vnir = R ./ overlap_vnir;
        coeff_overlap_swir = R ./ overlap_swir;
    end
    
    % Determine regions outside of the overlap
    ip_swir_to_vnir = interp1(wvl_swir,swir_spectrum,wvl_vnir);
    idx_no_overlap_at_vnir = isnan(ip_swir_to_vnir);
    no_overlap_wvl_vnir = wvl_vnir(idx_no_overlap_at_vnir);
    no_overlap_wvl_swir = wvl_swir(~idx_overlap_at_swir);
    vnir_no_overlap = vnir_spectrum(idx_no_overlap_at_vnir);
    swir_no_overlap = swir_spectrum(~idx_overlap_at_swir);
    
    % determine extreme coefficients phi_v and phi_s
    phi_vnir = coeff_overlap_vnir(1);
    phi_swir = coeff_overlap_swir(end);
    % Determine the window width to use in the logistic function later
    window_vnir = round(fit_window_vnir(delta_r));
    window_swir = round(fit_window_swir(delta_r));
    % Wavelengths affected by the correction
    affected_wvl_vnir = no_overlap_wvl_vnir(end-window_vnir+1:end);
    affected_wvl_swir = no_overlap_wvl_swir(1:window_swir);
    % Vector of final wavelengths
    wvl_full = [no_overlap_wvl_vnir;wvl_overlap;no_overlap_wvl_swir];
    % Central wavelength affected by the correction
    lambda_0_vnir = median(affected_wvl_vnir);
    lambda_0_swir = median(affected_wvl_swir);
    
    % Model the slope of the logistic function in VNIR
    k_vnir = fit_k_vnir(delta_r);
    % To model the slope of the logistic function in SWIR, use prior
    % information about the total number of bands outside of the
    % overlapping region
    k_swir = k_vnir*(length(no_overlap_wvl_vnir)/length(no_overlap_wvl_swir));
    
    % Implementation of logistic function to converge coefficients to 1
    % Sign function 
    if phi_vnir > 1
        L = 1;
    else
        L = -1;
    end
    % Logistic function
    coeff_vnir = L./(1+exp(-k_vnir*(no_overlap_wvl_vnir-lambda_0_vnir)))*(abs(phi_vnir-1))+1;
    coeff_swir = L./(1+exp(-k_swir*(no_overlap_wvl_swir-lambda_0_swir)))*(abs(phi_swir-1))+phi_swir;  
    % Final smooth spectrum
    full_spectrum = [coeff_vnir.*vnir_no_overlap;R;coeff_swir.*swir_no_overlap];

end