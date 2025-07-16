%
% AmFall_Github v1.0
%   - from PhAmFall v4.1
%   - Matlab code for AmFall paper (IEEE IOTJ)

%
% global parameter definitions
%
global Fs; % sampling frequency
global Ts; % sampling time
global Fc; % central carrier frequency
global Fsubs; % subcarrier spacing
global LightSpeed; % the speed of light
global Lambda; % the wavelength with Fc
global Gravity; % the Earth gravity acceleration

global Ncount; % the number of time indices
global Ntx;    % the number of Tx antennas
global Nrx;    % the number of Rx antennas
global Nsubc;  % the number of subcarriers in an antenna pair

% CSI measurement parameters
    Fs = 1000; % sampling frequency; Hz
    Fc = 5320; % carrier frequency; MHz
    Ts = 1/Fs; % sampling time

    % antenna and subcarrier parameters
    number_of_subcarriers = 30;

    Fsubs = 312.5; % K Hz sub-carrier spacing
    LightSpeed = physconst('LightSpeed');
    Gravity = 9.80665; % the earth gravity acceleration m/s^2
        
    Subc_20M = [ -28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1, ...
                            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28 ];
                    
    Subc_40M = [ -58, -54, -50, -46, -42, -38, -34, -30, -26, -22, -18, -14, -10, -6, -2, ...
                            2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58 ];

    Subc_Grp = Subc_20M;

    % for Lambda calculation
    Lambda = LightSpeed / (Fc * 10^6);
    Sc_freq = zeros (1, number_of_subcarriers);     % subcarrier frequencies
    Sc_lambda = zeros (1, number_of_subcarriers);   % subcarrier wavelenths
    Sc_k = zeros(1, number_of_subcarriers);         % subcarrier wavenumbers
                            
    for is = 1 : number_of_subcarriers
        Sc_freq (is) = Fc * 10^6 + Fsubs * 10^3 * Subc_Grp(is);
        Sc_lambda(is) = LightSpeed / Sc_freq(is);
        Sc_k (is) = 2*pi*Sc_freq(is) / LightSpeed;
    end % is
                            
    Sc_delta_f = Sc_freq - circshift(Sc_freq, 1); % vector
    Sc_delta_f(1) = 0;
                            
    Delta_f = Sc_freq(2) - Sc_freq(1);
    % redefine the Lambda
    Lambda = mean (Sc_lambda);

% moving variance parameters
    ws_sec = 0.1; % sec; window size for movmean() in sec; (2W+1) in (4) of the AmFall paper
    var_window_size_n = ws_sec * Fs;

% subcarrier selection parameters
    Select_n_ch = 30; % ell in (7) of AmFall paper;
    min_selected_ch = 5; % the minimum number of channels to be selected

% pc selection parameters
    pca_threhold = 0.01; % PCs less than pca_threshold regard as minor PCs (i.e., discarded)

% cwt & denoising parameters
    wavelet_name = "Morse";
    csi_voice_per_octave = 10;
    cwt_freq_lim = [ 1  170 ];
    cwt_max_clean_threshold = 1.8; % th_sc_{max} in AmFall paper (Algorithm 1)
    c_pl = 3; % (2/c_pl) is the kapa in AmFall paper (III-C-3)

% segmentation parameters for moving sum
    act_win_sec = 0.5; % sec; window for mvsum() to identify activity; (act_win_sec * 2 = 1.0 sec) is (2*W_m) in (13) of AmFall paper 
    act_window_size_n = act_win_sec * Fs;

% image generation color map
    color_map_n = 256; 

% CSI log file for input
csi_log_dir = "/Users/yongkeunkim/Documents/My Documents Local/SWJang/nlosfall_code/csi_log/";
csi_log_file = csi_log_dir + "envb/bh/fall/b_bh_fallfront3.log";
[~, csi_log_fn, ~] = fileparts(csi_log_file); % b_bh_fallfront3
% image file to save, mat file to save
csi_mat_file = "../test-csi/"+csi_log_fn ; % ex, b_bh_fallfront3.mat
csi_img_file = "../test-csi/"+csi_log_fn+".jpg"; % ex, b_bh_fallfront3.jpg
csi_log_file_name = split (csi_log_file, filesep); 
csi_log_file_name = csi_log_file_name{end}; % ex, b_bh_fallfront3.log

% time for segmentation
segment_time = 4; 
% image and mat file save flag; [image file, mat file]; 1 : save, 0 : no save
file_save_flag = [1, 1]; 
% plot supress flag, 0 : plotting, ~0 : supress plotting,
suppress_plot_flag = 0;  

% 
% end of AmFall parameters  %%%%%%%%%%%%%%%%%%
%

% constant definitions for execution
CSI_AMFALL_DATA = 1; 
CSI_SELECT = 2; 
CSI_PCA_EIG = 3;
CSI_CWT = 4;
CSI_SAVE_STEP = 5; % to save scalogram image upon the process

% execution sequence with control parameters
CMD_SEQ = [ ... % CMD, F1, F2, F3, F4, F5, F6, F7         
            CSI_AMFALL_DATA 0 0 0 0 0 0 0 ; 
            CSI_SELECT 0 0 0 0 0 0 0 ; 
            CSI_PCA_EIG 0 0 0 0 0 0 0;
            CSI_CWT 9  1 1 3 1 0 0; % no segment, normalize, denoising+vertical+horizontal(3), no eff check (speed), no eff apply 
            CSI_SAVE_STEP 0 0 0 0 0 0 0 ; 
];


for cmd = CMD_SEQ'

    switch cmd(1)        

        case CSI_AMFALL_DATA   
            % read csi_data from csi_log_file
            csi_trace = read_bf_file(csi_log_file);
            Ncount = length (csi_trace);
            csi_entry = csi_trace{100};
            csi = get_scaled_csi(csi_entry);
            Ntx = csi_entry.Ntx;
            Nrx = csi_entry.Nrx;
            Nsubc = size (csi_entry.csi, 3);

            Tx_idx = [ 1:Ntx ]; % Tx antenna indices; 
            Rx_idx = [ 1:Nrx ]; % Rx antenna indices;

            % let's make the ncount even for future fft()
            if mod (Ncount, 2) == 1 % odd number of ncount
                Ncount = Ncount - 1;
            end

            csi_data = zeros (Ncount, Ntx, Nrx, Nsubc, 'like', 1j);
            
            % read all the scaled data ncount * ntx * nrx * subc
            for i=1:Ncount
                csi_entry = csi_trace{i};
                csi = get_scaled_csi(csi_entry);
            
                i_ntx = csi_entry.Ntx;
                i_nrx = csi_entry.Nrx;
                i_nsubc = size (csi_entry.csi, 3);

                if i_nrx > Nrx 
                    i_nrx = Nrx;
                end
                
                for tx = 1 : i_ntx
                    for rx = 1 : i_nrx
                        csi_data (i, tx, rx, :) = csi (tx, rx, :);
                    end
                end
            end

            % use magnitude of csi_data
            csi_data = abs (csi_data);

        case CSI_SELECT % choose the best sensitive link-subc pair

            n_s = Select_n_ch; % ell in (7) of AmFall paper;  30 or 15

            [csi_selected, csi_table, vd_threshold] = csi_select5 (csi_data, ...
                        Tx_idx, Rx_idx, var_window_size_n, n_s);

            [cB, cI] = sort (csi_table(:, 1), "descend");
            csi_table_sort = csi_table (cI,:);

            csi_select_ratio = csi_table_sort(:,1) / sum (csi_table_sort(:,1));
            chosen_nchannels = length (csi_select_ratio(csi_select_ratio >= 1/n_s));

            % when it is too small, then, use the minimum channels 
            if chosen_nchannels < (n_s/3)
                chosen_nchannels = round (n_s/3);
            end
            chosen_nchannels = max (chosen_nchannels, min_selected_ch);

            if chosen_nchannels < n_s 
                rnd_selected_subc = sort (cI(1:chosen_nchannels), 'ascend');
                csi_selected = csi_selected (:, rnd_selected_subc);
            end
        
        case CSI_PCA_EIG % PCA processing         

            S = cov (csi_selected);
            [eV, eD] = eig(S);
            [sB, sI] = sort (diag(eD), "descend");
            eD = eD(sI, sI);
            eV = eV(:,sI);
            csi_pc = csi_selected * eV;

            norm_eigen = diag(eD) / sum (diag(eD));
        
            Npc = length (norm_eigen( norm_eigen >= pca_threhold));

            csi_pc_sel_idx = pc_select5 (csi_pc, eD, pca_threhold, var_window_size_n);

            csi_pca_sum = sum (csi_pc (:, csi_pc_sel_idx), 2);

        case CSI_CWT % CWT image generation

            n_fig = cmd (2);
            if suppress_plot_flag
                n_fig = 0;
            end
            cwt_seg_flag = cmd (3); % 0 : no segmantation, 1 : segmentation
            cwt_normalize_flag = cmd (4); % 0 : no normalize, 1 : normalize
            cwt_denoising_flag = cmd (5); % 0 : no denoising, 1 : denoising
            cwt_eff_freq_flag = cmd (6); % 0 : no effective frequency checking, 1 : effective frequency checking 

            fb = cwtfilterbank (wavelet=wavelet_name, SignalLength=Ncount,...
                FrequencyLimits=cwt_freq_lim, ...
                VoicesPerOctave=csi_voice_per_octave, ... % 10 for SNU DS, 12 for Jordan DS
                SamplingFrequency=Fs);         

            % cwt
            [csi_cfs, ff] = wt(fb, csi_pca_sum);
            [ ~, cwt_signal_quality] = signal_quality(csi_pca_sum, var_window_size_n);

            cfs_s = size (csi_cfs);

            if cwt_normalize_flag
                csi_cfs = csi_cfs ./ max (csi_cfs(:));
                if n_fig % plotting; 
                    plot_scalogram ( abs(squeeze(csi_cfs)), n_fig, 1, [Fs:Fs:cfs_s(2)], [1:1: (round(cfs_s(2)/Fs))], [5:10:length(ff)],  floor( ff([5:10:length(ff)])), ...
                       sprintf("Normalized cfs : %s\n signal quality : %s", csi_log_file_name, num2str(cwt_signal_quality)) )
                end
            end

            cwt_eff_max_freq = zeros (1, Ncount); 
            cwt_eff_freq = zeros (1, Ncount);
            if cwt_denoising_flag

                % general denoising; Algorithm 1
                clean_threshold = min (1 / log10(cwt_signal_quality), cwt_max_clean_threshold); % 1.8

                freq_mean = mean (abs(csi_cfs), 2) * clean_threshold;
                freq_mean_mat = abs(csi_cfs) - repmat(freq_mean, 1, size(csi_cfs,2));
                csi_cfs_ind = freq_mean_mat > 0;

                csi_cfs = csi_cfs .* csi_cfs_ind;

                if n_fig % plotting
                    plot_scalogram ( abs(squeeze(csi_cfs)), n_fig, 2, [Fs:Fs:cfs_s(2)], [1:1: (round(cfs_s(2)/Fs))], ...
                        [5:10:length(ff)], floor(ff([5:10:length(ff)])), ["General denoising : "+csi_log_file_name] )
                end

                if cwt_denoising_flag >= 2 % + vertical denoising
                    csi_cfs_ind = abs(csi_cfs) > 0;
                    cwt_high_freq = csi_cfs_ind .* repmat(ff, 1, size(csi_cfs_ind,2));
                    [cwt_eff_freq ] = cfs_vertical_denoise (cwt_high_freq, (ff(1) - ff(2)) ); 
                    csi_cfs_ind = (cwt_eff_freq ~= 0); 
                    csi_cfs = csi_cfs_ind .* csi_cfs; 
                    
                    if n_fig % plotting
                        plot_scalogram ( abs(squeeze(csi_cfs)), n_fig, 3, [Fs:Fs:cfs_s(2)], [1:1: (round(cfs_s(2)/Fs))], ...
                            [5:10:length(ff)], floor(ff([5:10:length(ff)])), ["Vertical denoising : "+csi_log_file_name] )
                    end
                end % 

                if cwt_denoising_flag >= 3 % + horizontal denoising
                    % horizontal denoising
                    ff_t = ff * (Lambda/Gravity)*(c_pl/2);
                    ff_dt = ff_t - circshift(ff_t, -1); ff_dt(end)=0;
                    
                    csi_cfs = cfs_horizontal_denoise (csi_cfs, ff_dt * Fs );
 
                    if n_fig % plotting
                        plot_scalogram ( abs(squeeze(csi_cfs)), n_fig, 4, [Fs:Fs:cfs_s(2)], [1:1: (round(cfs_s(2)/Fs))], ...
                            [5:10:length(ff)], floor(ff([5:10:length(ff)])), ["Horizontal denosing : "+csi_log_file_name] )
                    end
                end % cwt_denoising_flag >= 3 

                if cwt_eff_freq_flag % speed info from effective freq checking

                    img_cfs_ind = abs(csi_cfs) > 0;
                    img_cfs_mean = sum (abs(csi_cfs(:))) / sum (img_cfs_ind(:));      
                    img_cfs2 = abs(csi_cfs) - img_cfs_mean ;

                    img_cfs_ind2 = img_cfs2 > 0;

                    cwt_high_freq = img_cfs_ind2 .* repmat(ff, 1, size(csi_cfs,2));
                    [cwt_eff_freq ] = cfs_vertical_denoise (cwt_high_freq, (ff(1) - ff(2)) );
                    
                    if n_fig % plotting
                        plot_scalogram ( abs(squeeze((cwt_eff_freq ~= 0).* csi_cfs)), n_fig, 5, [Fs:Fs:cfs_s(2)], [1:1: (round(cfs_s(2)/Fs))], ...
                            [5:10:length(ff)], floor(ff([5:10:length(ff)])), ["cfs for speed estimation : "+csi_log_file_name] )
                    end % n_fig
    
                    cwt_eff_max_freq = max(cwt_eff_freq, [], 1);
            
                    delta_t = Fs/2; % 0.5 second for acceleration

                    csi_speed = cwt_eff_max_freq * Lambda;

                    csi_acceleration = (csi_speed - circshift(csi_speed, delta_t)) / (Ts*delta_t);

                end % cwt_eff_freq_flag

            end % cwt_denoising_flag

            if cwt_seg_flag % segmentation

                [activity_locs, ms, max_msi, peak_prominence] = find_activity (csi_pca_sum, var_window_size_n, act_window_size_n);

                % when the clean_threshold is big, the moving sum variance is not reliable
                % in this case, let's use the max frequency point as a center of the activity if it is very noisy 
                if cwt_signal_quality <= 2 % 
                    activity_locs = [];
                    [ ~, max_msi] = max (cwt_eff_max_freq);
                end 

                switch length (activity_locs)
                    case 0 % no peak
                        m_time = max_msi;
                    case 1 % one peak
                        m_time = activity_locs(1);
                    otherwise % more than 2 peaks
                        m_time = 0;
                end
                [m_time_s,  m_time_e] = csi_segment3 (activity_locs, m_time, cwt_eff_max_freq, segment_time*Fs, act_window_size_n, Ncount);
 
            else 
                m_time_s = 1
                m_time_e = Ncount;
            end 

            h_idx = length (ff (ff > cwt_freq_lim (2)) );
            if h_idx == 0 % PhAmFall 4.0.2
                h_idx = 1;
            end
            l_idx = length (ff (ff > cwt_freq_lim (1)) );

             if n_fig && cwt_seg_flag % plotting
                 plot_scalogram ( abs(squeeze(csi_cfs)), n_fig, 6, [Fs:Fs:cfs_s(2)], [1:1: (round(cfs_s(2)/Fs))], ...
                            [5:10:length(ff)], floor(ff([5:10:length(ff)])), sprintf("%s ", csi_log_file_name) )
                 % draw white dot line for segmention period
                 hold on
                 % draw the effect period (in x-axis) 
                 line([m_time_s,m_time_s], [cfs_s(1), 1], 'Color', 'w', 'LineStyle','--');
                 line([m_time_e,m_time_e], [cfs_s(1), 1], 'Color', 'w', 'LineStyle','--');
                 % draw the effect frequency band (in y-axis)
                 line([1,cfs_s(2)], [h_idx, h_idx], 'Color', 'w', 'LineStyle','--');
                 line([1,cfs_s(2)], [l_idx, l_idx], 'Color', 'w', 'LineStyle','--');
                 hold off
             end
            if cwt_seg_flag 
                 csi_cfs_denoised_noseg = csi_cfs;
                 csi_cfs = csi_cfs (h_idx:l_idx, m_time_s:m_time_e); % complex data
                 csi_speed = csi_speed (m_time_s:m_time_e); % real data
                 csi_acceleration = csi_acceleration (m_time_s:m_time_e); % real data

                 if n_fig % plotting
                    plot_scalogram ( abs(squeeze(csi_cfs)), n_fig, 7, [Fs:Fs:cfs_s(2)], [1:1: (round(cfs_s(2)/Fs))], ...
                        [5:10:length(ff)], floor(ff([5:10:length(ff)])), ["Segmented cfs : "+csi_log_file_name] )

                    im = ind2rgb(round(rescale(abs(squeeze(csi_cfs)),0,255)),jet(color_map_n));
                    image_yratio = 224/ size (ff,1);
                    image_xratio = 224 / segment_time;
                    plot_scalogram ( imresize(im,[224 224]), n_fig, 8, [image_xratio:image_xratio:segment_time*image_xratio], [1:1: (round(cfs_s(2)/Fs))], ...
                        [5*image_yratio:10*image_yratio:length(ff)*image_yratio], floor(ff([5:10:length(ff)])), ["Generated image : "+csi_log_file_name] )
                 end
            end

        case CSI_SAVE_STEP

            if file_save_flag(2) % save mat file
                save (csi_mat_file, "csi_cfs", "csi_speed", "csi_acceleration", "cwt_signal_quality");
            end

            if file_save_flag(1) % save jpg (image) file
                im = ind2rgb(round(rescale(abs(squeeze(csi_cfs)),0,255)),jet(color_map_n));
                imwrite(imresize(im,[224 224]),csi_img_file);
            end

    end % swtich
end % for cmd


% plot_scalogram
function plot_scalogram (x, n_fig, n_sp, x_tick, x_label, y_tick, y_label, title_str)
    hf = figure (n_fig); 
    hf.Color='w';   
    subplot (3,3,n_sp);
    imagesc(x)
    %colorbar;
    set(gca, 'YTick', y_tick) %[1 5 15 25 35 45 55 65 75] )
    set(gca, 'YTickLabel', y_label )
    set(gca, 'XTick', x_tick)
    set(gca, 'XTickLabel', x_label) % sec ticks
    xlabel("Time (s)",'fontsize', 18)
    ylabel("Frequency (Hz)",'fontsize',18)
    title( title_str, 'fontsize',18)
end


% CSI Stream Selection process function
function [csi_selected, csi_table, th] = csi_select5 (x, Tx, Rx, ws, n_s)

    if size (x,2) == 1 % Ntx == 1
        Tx = 1;
    end
    if size (x,3) == 1 % Nrx == 1
        Rx = 1;
    end

    x = x(:, Tx, Rx, :);

    n_c = size(x,1);
    n_tx = size(x,2);
    n_rx = size(x,3);
    n_subc = size(x, 4); 

    [~, csi_var_diff] = signal_quality(x, ws);

    % to get the threshold value
    [vB,  ~] = sort (csi_var_diff(:), "descend");
    th = vB(n_s);

    csi_selected = zeros (n_c, n_s);
    csi_table = zeros (n_subc, 4);
    % take all subcarriers above the threshold; it is not ORDERED
    ic = 1;
    for it = 1 : n_tx
        for ir = 1 : n_rx
            for is = 1 : n_subc
                if csi_var_diff (1,it, ir, is) >= th && ic <= n_s
                     csi_selected (:, ic)= x(:, it, ir, is);
                     csi_table (ic, 1) = csi_var_diff (1,it, ir, is);
                     csi_table(ic,2) = it;
                     csi_table (ic, 3) = ir;
                     csi_table (ic, 4) = is;
                     ic = ic + 1;
                 end
            end % Nsubc
        end % Nrx
    end % Ntx

end % csi_select()


% Eq.(6) of AmFall paper
function [ x_var, x_quality] = signal_quality (x, ws)
    x_var = movvar(x, ws, 0, 1);
    x_quality =  max(x_var, [],1) ./ mean (x_var, 1);
end


% PC Selection process function
function [ pc_sel ] = pc_select5 (x, ev, pc_th, ws)

    % firstly check the portion of 1st PC
    norm_eigen = diag(ev) / sum (diag(ev));
    c_npc = length (norm_eigen( norm_eigen >= pc_th)); % candidate no of pc

    % let's make decision for minimum 3 PCs
    while c_npc  <= 2
        pc_th = pc_th / 3;
        c_npc = length (norm_eigen( norm_eigen >= pc_th));
    end
    [ ~, pc_var_diff ] = signal_quality(x(:, 1:c_npc), ws);

    [~, cI] = sort (pc_var_diff, "descend");

    pc_select_ratio = pc_var_diff / sum (pc_var_diff);
    chosen_pcchannels = length (pc_select_ratio(pc_select_ratio >= 1/(c_npc-1)));
    if chosen_pcchannels < 1
         chosen_pcchannels = 1;
    end
    pc_sel = cI (1: chosen_pcchannels);
    pc_sel = sort (pc_sel, 'ascend');
end


% vertical denoising; Algorithm 2 of AmFall paper
function [cwt_high_freq ] = cfs_vertical_denoise (cwt_high_freq, lf)

    global Gravity;
    global Lambda;
    global Ts;

    dfs_p_sample = Gravity * Ts * (1/Lambda) ; % 0.163;
    ws_ev = ceil (lf / dfs_p_sample) ; % 35
    cef = 0; % let's start the initial cef as zero % the current effective frequency
    for iw = 1 : size (cwt_high_freq,2) %  Ncount
        [ ncf, ncf_i ] = max(cwt_high_freq (:, iw)); % next highest frequency       
        while ncf 
            if ncf <= cef % consider as effective
                break;
            end           
            delta_f  = ncf - cef;
            delta_n = round (delta_f / dfs_p_sample); % required window size 
            ws_s = max ((iw-1-delta_n), 1);
            if delta_n <= ws_ev   
                a_cev = mean ( max (cwt_high_freq(:, ws_s : (iw-1)), [], 1)); % the reference cev        
                if (a_cev * 0.9) <= cef && (a_cev * 1.1)  >= cef 
                    break;
                end
            end % if delta_n
            cwt_high_freq(ncf_i, iw) = 0;
            [ ncf, ncf_i ] = max(cwt_high_freq (:, iw));  % find the next highest
        end % while
        cef = ncf;
    end % for
end


% horizontal denoising; Algorithm 3 of AmFall paper
function [csi_cfs_clean] = cfs_horizontal_denoise (csi_cfs, n_ff_dt)
           
    csi_cfs_clean = csi_cfs;
    for i = 1 : numel (n_ff_dt) % each frequency
        one_count = 0;
        for j = 1 : size (csi_cfs_clean, 2) % time index
            if abs (csi_cfs_clean(i, j) )
                one_count = one_count + 1;
            else
                if one_count == 0
                    continue;
                end
                if one_count < n_ff_dt(i)
                    csi_cfs_clean(i, j-one_count : j-1) = 0;
                    one_count = 0;
                end                  
            end
        end
    end
end


% find the number of activities in the scalogram
function [activity_locs, ms, max_msi, peak_prominence] = find_activity (csi_pca_sum, var_window_size_n, act_window_size_n)

    act_win_size = act_window_size_n;

    ms = movsum (movvar(csi_pca_sum, var_window_size_n), [act_win_size act_win_size], 1);

    [ max_ms, max_msi ] = max(ms);
    peak_prom_ratio = 0.2;
    peak_prominence = peak_prom_ratio * max_ms;

    [~, activity_locs] = findpeaks (ms, 'MinPeakProminence', peak_prominence);

end


% a part of III-D of AmFall paper for segmentation
function [m_time_s, m_time_e] = csi_segment3 (activity_locs, m_time, cwt_eff_max_freq, segment_n, act_win_size, Ncount)

    tw_start = round(segment_n/2);
    tw_end = round (segment_n/2);

    if m_time == 0 % need to calculate m_time
        selected_seg = 1;
        seg_max = -Inf;

        for ia = 1 : length (activity_locs)
            start_idx = activity_locs(ia) - act_win_size;
            end_idx = activity_locs(ia) + act_win_size;
            if start_idx < 1
                start_idx = 1;
            end
            if end_idx > Ncount
                end_idx = Ncount;
            end       
            if seg_max < max ( cwt_eff_max_freq (start_idx:end_idx))
                 seg_max = max ( cwt_eff_max_freq (start_idx:end_idx));
                 selected_seg = ia;
            end
        end % for ia
        m_time = activity_locs(selected_seg);
    end % if length (activity_locs) > 1 

    m_time_s = m_time - (tw_start);
    m_time_e = m_time + (tw_end);
    % when the start time is less than 1
    if (m_time_s <= 0)
          m_time_e = m_time_e + abs (m_time_s) + 1;
          m_time_s = 1;
    end
    % when the end time is beyond Ncount
    if (m_time_e > Ncount)
          m_time_s = m_time_s - (m_time_e - Ncount);
          if m_time_s <= 0
               m_time_s = 1;
          end
          m_time_e = Ncount;
    end
    m_time_e = m_time_e - 1; % considering the fixed size, ex, 3000 instead of 3001
end

