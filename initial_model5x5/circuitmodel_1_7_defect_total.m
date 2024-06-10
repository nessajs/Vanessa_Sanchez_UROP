%This script looks at current distribution within a 5 tape x 5 resistor
%circuit model with a defect. The input is the slice of the circuit that we want to look
%at and the placement of the defect. It takes that slice and plots the current distribution among the tapes for each total
%current value in the range (100,800).


%what slice to look at:
slice = 3;

%% Independent sources
runs = 8;
I_total_set = linspace(100,800,runs);

%for each run, take the converged values (store them first) or the last values
%the while loop stops at these conditions, so store values coming out of loop
%before: plot evolution of each resistor (column) for every tape
%now: plot evolution of last values of each resistor for each of the rows

converged = {}; %array of arrays
for run = I_total_set
    I_total = run;
    
    global I_val
    I_val = I_total;
    
    %% Material & Geometric Setup taken from Richard's code
    N_x = 5; 
    N_HTS_tapes = 5; % M
    L_cable = 50;               % Length of cable (cm)
    L_R_long = L_cable/N_x;     % Length of longitudinal resistors (cm)
    
    n_index = 29.8;         % Power law index for one HTS tape
    Ic = 128.4;             % Critical current (A) of one HTS tape  , B dependent
    Ec = 1;                 % Quench electric field (µV/cm)
    Vc = Ec*L_R_long;       % Quench criterion in each resistor (µV)
    
    rho_Cu = 0.1956;        % Resistivity of copper at 77 K (µΩ-cm)
    rho_PbSn = 2.84;        % Resisitivity of Pb27Sn63 at 77 K (µΩ-cm) ...
                            
    rho_Ag = 0.2694;        % Resistivity of silver at 77 K (µΩ-cm)
    rho_Hastelloy = 124.0;  % Resistivity of Hastelloy at 77 K (µΩ-cm)
    t_groove = 0.060*2.54;  % Height of tape groove (inch -> cm)
    t_cable = 0.551*2.54;   % Total height of cable (inch -> cm)
    t_former = (t_cable - t_groove)/2;      % Thickness of Cu former (top or bottom half)
    N_Cu_tapes_top = 1;     % Number of Cu filler tapes above HTS stack
    N_Cu_tapes_bottom = 1;  % Number of Cu filler tapes below HTS stack
    t_HTS_tape = 56e-4;     % Thickness of HTS tape (cm)
    t_HTS_layer = 2e-4;     % Thickness of HTS layer (cm)
    t_Cu_tape = 4.5*25.4e-4;  % Thickness of Cu filler tape (thou -> cm)
    t_Cu_layer_thin_side = 5e-4;     % Thickness of Cu layer, thin side of HTS tape (cm)
    t_Ag_layer_thin_side = 3e-4;     % Thickness of Ag layer, thin side of HTS tape (cm)
    t_Hastelloy_thick_side = 40e-4;  % Thickness Hastelloy, thick side of HTS tape (cm)
    t_Ag_layer_thick_side = 1e-4;    % Thickness of Ag layer, thick side of HTS tape (cm)
    t_Cu_layer_thick_side = 5e-4;    % Thickness of Cu layer, thick side of HTS tape (cm)
    t_PbSn = (t_groove - (N_HTS_tapes*t_HTS_tape + (N_Cu_tapes_top + N_Cu_tapes_bottom)*t_Cu_tape))/(N_HTS_tapes + N_Cu_tapes_top + N_Cu_tapes_bottom); % Average thickness of PbSn between tapes (cm)
    d_cable = 0.4;         % Depth of the cable (cm)
    
    % Thick/thin layer resistances
    R_thin = (rho_Cu*t_Cu_layer_thin_side + rho_Ag*t_Ag_layer_thin_side)...
        /(L_R_long*d_cable);
    R_thick = (rho_Cu*t_Cu_layer_thick_side + rho_Ag*t_Ag_layer_thick_side + ...
        rho_Hastelloy*t_Hastelloy_thick_side)/(L_R_long*d_cable);
    R_thin_thin = R_thin*2;
    R_thick_thick = R_thick*2;
    R_thin_thick = R_thin + R_thick;
    
    % Thicknesses of these layers:
    t_thin = t_Cu_layer_thin_side + t_Ag_layer_thin_side;
    t_thick = t_Cu_layer_thick_side + t_Ag_layer_thick_side + t_Hastelloy_thick_side;
    t_thin_thin = t_thin*2;
    t_thick_thick = t_thick*2;
    t_thin_thick = t_thin + t_thick;
    
    %% Longitudinal Resistances
    
    % Copper former resistance
    global R_Cu_val
    R_Cu_val = rho_Cu*L_R_long/(d_cable*t_former);
    
    % Initial HTS resistance to be assigned
    R_hts0 = R_Cu_val/10; %rho_Cu*L_R_long/(d_cable*t_HTS_layer);
    
    global N
    global M
    N = 5; %5 slices
    M = 5; %5 tapes
    
    global R
    global R_defect
    global R_inf
    R_inf = 10; 
    
    R = repmat(R_hts0, M, N); %initialize hts matrix 5x5
    %initialize defect placeholder at the defect location
    R_defect = R_inf;
    R(1,3) = R_defect; %make it easier to change this, maybe don't hardcode the initialization of R
    
    Error0 = 50;
    E_slice = repmat(Error0, M, N); %initialize a corresponding error matrix
    E = E_slice;
    
    Error0_compute = 10;
    E_compute_slice = repmat(Error0_compute, M, N);
    E_compute = E_compute_slice;
    
    I_compute_matrix_slice = repmat(Error0_compute, M, N);
    I_compute_matrix = I_compute_matrix_slice;
    
    former_slice = repmat(Error0_compute, 2, N);
    former = former_slice;
    %% Transverse Resistors
    
    % Padding resistance
    global R_Padding_val
    R_Padding_val = R_thin + (rho_Cu*(t_former/2 + N_Cu_tapes_top*t_Cu_tape) + rho_PbSn*(N_Cu_tapes_top + 1)*t_PbSn)/(L_R_long*d_cable);
    
    % Intra HTS Resistances
    global R_Intra_HTS23_val
    global R_Intra_HTS34_val
    global R_Intra_HTS45_val
    global R_Intra_HTS56_val
    
    R_Intra_HTS23_val =  R_thick_thick+ rho_PbSn*t_PbSn/(L_R_long*d_cable);
    R_Intra_HTS34_val =  R_thin_thick + rho_PbSn*t_PbSn/(L_R_long*d_cable);
    R_Intra_HTS45_val =  R_thin_thin + rho_PbSn*t_PbSn/(L_R_long*d_cable);
    R_Intra_HTS56_val =  R_thick_thick + rho_PbSn*t_PbSn/(L_R_long*d_cable);
    
    fprintf('\nStarted -- please be patient.\n\n');
    
    %% Print out the netlist 
    
    fprintf('Netlist:');
    
    fname = "C:\Users\Vaness Sanchez\Desktop\scam-master\no_block_defect_circuit.txt"; %replace with path to .txt file
    type(fname) % prints our netlist
    fid = fopen(fname); 
    fileIn=textscan(fid,'%s %s %s %s %s %s');  
    % Split each line into 6 columns, the first item is always the name of the
    % circuit element and the second and third items are always node numbers
    
    global arg3
    global Name
    global N1
    global N2
    [Name,  N1, N2, arg3, arg4, arg5] = fileIn{:};
    N1 = str2double(N1);
    N2 = str2double(N2);
    
    fclose(fid);
    global nLines
    nLines = length(Name);  % Number of lines in file (or elements in circuit).
    
    %% Find Resistance Indices for ease of updating resistances
    
    %arg3 holds resistors, store them in ref_value so we can reference later
    global ref_value 
    ref_value = arg3; 
    
    global ref_idx
    global ref_hts
    % to use when looking through netlist
    ref_hts = {'R_hts1', 'R_hts2', 'R_hts3', 'R_hts4', 'R_hts5'}; %go through this list
    ref_idx = []; %find the index where it first occurs
    
    % Find all indices in netlist where we have a new HTS tape and store
    for i=1:length(ref_hts)
        idx = find(strcmp(arg3,ref_hts(i)),1);
        ref_idx = [ref_idx, idx];
    end
    
    % now we're able to replace values in ref_value bc we know the indices (ref_value is what's fed into the fcn)
    for idx=1:length(ref_hts)
        col = 0;
        
        % entering initial R vals from R matrix
        for good_idx=ref_idx(idx):ref_idx(idx)+(N-1)
            col=col+1;
            if ~strcmp(ref_value{good_idx}, 'R_defect1')
                ref_value(good_idx) = num2cell(R(idx,col)); 
            end
        end
    end
    
    
    %% Set up convergence check
    
    % initialize a matrix to converged = false
    epsilons = false(N,M);
    count = 0;
    iter = 0;
    
    I_hts_old = repmat(300, M, N);
    I_compute_old = repmat(300, M, N);
    %iterations = 70;
    iter = 0;

    % all returns 1 if all elements nonzero/logical, so this iterates while not entirely converged
    while not(all(all(epsilons)))    
    %for iter = 0:iterations %use line above instead if wanting to run until convergence
        iter = iter + 1;
        
        epsilons = false(N,M); %clear every time bc we care about all tapes converging at once
        
        %% Find initial values
        if iter == 1
            %we need to calculate initial deltas because we cant update resistances without them
    
            x = ComputeFunction(); %uses R with initial values
    
            %% set up tape config
            gnd = 0;
                        
            hts1 = [x(8),x(9),x(10),x(11),x(12),x(13)]; %6 nodes, 5 resistors
            hts2 = [x(14),x(15),x(16),x(17),x(18),x(19)];
            hts3 = [x(20),x(21),x(22),x(23),x(24),x(25)];
            hts4 = [x(26),x(27),x(28),x(29),x(30),x(31)];
            hts5 = [x(32),x(33),x(34),x(35),x(36),x(37)];
    
            % former
            former_nodes  = [x(2),x(3),x(4),x(5),x(6),x(7); x(38),x(39),x(40),x(41),x(42),x(43)];
                    
            % make a matrix (not the most efficient lol)
            hts_tapes = [];
            hts_tapes = [hts_tapes; hts1];
            hts_tapes = [hts_tapes; hts2];
            hts_tapes = [hts_tapes; hts3];
            hts_tapes = [hts_tapes; hts4];
            hts_tapes = [hts_tapes; hts5];
       
            former_v = [];
            for r = 1:2
                delta_r = [];
                row = former_nodes(r,:);
                for c = 2:length(row) %go through length of tape
                    delta = row(c-1) - row(c); %high to low potential
                    delta_r(end+1) = abs(delta); % positive number for voltage drop - check if necessary
                end
                former_v(r,:) = delta_r;
            end
            former_slice = former_v/R_Cu_val; %now we have currents through former
            former(:,:,iter) = former_slice;
    
    
            % take initial differences between node voltages
            deltaVs = [];
            for id = 1:length(hts_tapes(:,1)) %for the 5 tapes
                deltaV = [];
                tape = hts_tapes(id,:);
                for jd = 2:length(tape) %go through length of tape
                    delta = tape(jd-1) - tape(jd); %high to low potential
                    deltaV(end+1) = abs(delta); % positive number for voltage drop - check if necessary
                end
                deltaVs = [deltaVs; deltaV];
            end
       
        
        else
            %now we have deltas we can use to start updating Rs, we will find all subsequent deltas in this else statement 
            
            %(i,j) = (row, column)
            tape_num = N;    
            for i = 1:tape_num % for each tape    
                % update all my hts resistances, checking once if each has converged
    
                slice_num = M; %5 slices
                for j = 1:slice_num % go through all the slices  
                       
                        fprintf('i = %d, j = %d\n', i, j);               
        
                        delta = deltaVs(i,j);
                        R_old = R(i,j); %for the defect, R_old will always be R_inf
                        %R is used for conv check, then is replaced with R_new      
    
                        % use Ohm's law to calculate current through that resistor, I_compute
                        I_compute = delta/R_old; % V = IR
                        I_compute_matrix_slice(i,j) = I_compute; % will be computed for defect as well
            
                        I_hts = Ic*(delta/Vc)^(1/n_index); % I = Ic*(V/Vc)^1/n
                
                        % just a check for bad values
                        if isnan(I_hts) || isinf(I_hts)
                            fprintf('I_hts is NaN or Inf, i = %d, j = %d\n',i,j)
                            return
                        end
    
                        if isnan(R_old) || isinf(R_old)
                            fprintf('R_old is NaN or Inf, i = %d, j = %d\n',i,j)
                            return
                        end
                    
                        if R(i,j) ~= R_inf %update unless it's a defect                                
    
                            % find new hts R
                            R_new = R_old*(I_compute/I_hts);
                           
                            %% check convergence
                            R_small = 1*10^-6;
        
                            % update R 
                            if R_new < R_small 
                                R(i,j) = R_small;
                                display('R_small set')
        
                            else
                                R(i,j) = R_new;
                     
                            end
                        end
       
    
                        % check if converged
                        if (abs((I_compute_old(i,j) - I_compute)/I_compute_old(i,j)) < 10^-3) | (iter == 500)| (R(i,j) == R_inf) %after finding that I = 200A does not converge with the given criteria, doing this
                            epsilons(i,j) = true;
                        end                
    
                        % compute percentage error for convergence analysis
                        percent_error = (I_hts_old(i,j) - I_compute)/I_hts_old(i,j) * 100;
                        E_slice(i,j) = percent_error;
                        %fprintf('error: %f %% \n ', percent_error)
    
                        % another way to look at error (I_compute vs I_compute_old)
                        compute_percent_error = (I_compute_old(i,j) - I_compute)/I_compute_old(i,j) * 100;
                        E_compute_slice(i,j) = compute_percent_error;
    
                        I_hts_old(i,j) = I_hts;
                        I_compute_old(i,j) = I_compute;
      
                end
         
            end
            I_compute_matrix(:, :, iter) = I_compute_matrix_slice;
            E(:, :, iter) = E_slice;
            E_compute(:, :, iter) = E_compute_slice;
            %generate heatmap for better analysis of error 
            %heatmap(E(:,:,2)) %specify iteration 
    
            %now we have iterated through all resistors, go in and replace ref_value with R matrix values which were updated above
            for idx=1:length(ref_hts)
                col=0;
                for good_idx=ref_idx(idx):ref_idx(idx)+(N-1)
                    col=col+1;
                    if ~strcmp(ref_value{good_idx}, 'R_defect1') %excluding defect
                        ref_value(good_idx) = num2cell(R(idx,col)); 
                    end
                end
            end
        
            %% Call function for new node voltages
            x = ComputeFunction(); %returns x
            realX = x;
    
            %% organize target results
    
            gnd = 0;
                        
            %pad1 = [x(1),x(2),x(3),x(4),x(5),x(6),gnd]; %still calculated just not tracked
            hts1 = [x(8),x(9),x(10),x(11),x(12),x(13)]; 
            hts2 = [x(14),x(15),x(16),x(17),x(18),x(19)];
            hts3 = [x(20),x(21),x(22),realX(23),realX(24),x(25)];
            hts4 = [x(26),x(27),x(28),x(29),x(30),x(31)];
            hts5 = [x(32),x(33),x(34),x(35),x(36),x(37)];
            %pad2 = [v_1, v_32,v_33,v_34,v_35,v_36,gnd];
            
            %former
            former_nodes  = [x(2),x(3),x(4),x(5),x(6),x(7); x(38),x(39),x(40),x(41),x(42),x(43)];
    
    
            hts_tapes = [];
            hts_tapes = [hts_tapes; hts1];
            hts_tapes = [hts_tapes; hts2];
            hts_tapes = [hts_tapes; hts3];
            hts_tapes = [hts_tapes; hts4];
            hts_tapes = [hts_tapes; hts5];
    
            %% recalculate deltas
            former_v = [];
            for r = 1:2
                delta_r = [];
                row = former_nodes(r,:);
                for c = 2:length(row) %go through length of tape
                    delta = row(c-1) - row(c); %high to low potential
                    delta_r(end+1) = abs(delta); % positive number for voltage drop - check if necessary
                end
                former_v(r,:) = delta_r;
            end
            former_slice = former_v/R_Cu_val; %now we have currents through former
            former(:,:,iter) = former_slice;
    
            deltaVs = [];
            for id = 1:length(hts_tapes(:,1))
                deltaV = [];
                tape = hts_tapes(id,:);
                for jd = 2:length(tape)
                    delta = tape(jd-1) - tape(jd);
                    deltaV(end+1) = abs(delta); % positive number for voltage drop
                end
                deltaVs = [deltaVs; deltaV];   
            end
    
        end
    
        %% current through copper -- not necessary rn
        cu1_delta = x(1) - x(2); %top,left
        cu2_delta = x(1) - x(38); %bottom,left
    
        cu3_delta = x(7); %top,right
        cu4_delta = x(43); %bottom,right
    
        cu1_I = cu1_delta/R_Cu_val;
        cu2_I = cu1_delta/R_Cu_val;
        cu3_I = cu1_delta/R_Cu_val;
        cu4_I = cu1_delta/R_Cu_val;

    
        %% values to display
        R;
        deltaVs;
        epsilons;
        hts_tapes; %nodal voltages

    
    end
    
    converged{end + 1} = I_compute_matrix(:,slice,end);

end

clf
figure (4)
hold on; % This command ensures that each new line is added to the same plot
for i = 1:5
    rowdata = [];
    for j = 1:length(converged)
        data = converged{j}(i); % Extract data from the ith cell
        rowdata(end + 1) = data;
    end
    plot(I_total_set, rowdata, 'o-');    % Plot the data
    legend('Tape 1', 'Tape 2', 'Tape 3', 'Tape 4', 'Tape 5');
end
hold off;


%% define function
function x = ComputeFunction()
%% Substitute
    global R
    global R_inf
    global ref_idx
    global ref_hts
    global ref_value
    global N
    global nLines
    global Name
    global N1
    global N2
    global arg3
    global R_Cu_val
    global R_Padding_val
    global I_val

    global R_Intra_HTS23_val
    global R_Intra_HTS34_val
    global R_Intra_HTS45_val
    global R_Intra_HTS56_val

    Value = ref_value; %have been initiated w initial R values

    % now there's only the need to substitute non-hts resistances
    for ix=1:nLines
        val = char(arg3(ix));
        switch val
            case 'R_cu'
                val = R_Cu_val;
                Value{ix} = val;
            case {'R_inf' , 'R_defect1'}
                val = R_inf;
                Value{ix} = val;
            case 'R_Padding'
                val = R_Padding_val;
                Value{ix} = val;
            case 'R_Intra_HTS23'
                val = R_Intra_HTS23_val;
                Value{ix} = val;
            case 'R_Intra_HTS34'
                val = R_Intra_HTS34_val;
                Value{ix} = val;
            case 'R_Intra_HTS45'
                val = R_Intra_HTS45_val;
                Value{ix} = val;
            case 'R_Intra_HTS56'
                val = R_Intra_HTS56_val;
                Value{ix} = val;
            case 'I'
                val = I_val;
                Value{ix} = val;
            otherwise('Unexpected Component');
            %display(val)
        end
    end

    Value=Value.';
    %Value_double = str2sym(arg3);
    Value;
    
    
    
    n = max([N1; N2]);   % Find highest node number (i.e., number of nodes)
    
%     %NEXT 7 LINES NOT USED RN, we have no voltage sources
     m=0; % "m" is the number of voltage sources, determined below.
%     for k1=1:nLines                  % Check all lines to find voltage sources
%         switch Name{k1}(1)
%             case {'V', 'O', 'E', 'H'}  % These are the circuit elements with
%                 m = m + 1;             % We have voltage source, increment m.
%         end
%     end
    
    % Preallocate all arrays (use Litovski's notation).hts_tapes = [hts_tapes; hts1];
    G=cell(n,n);  [G{:}]=deal(0);    % G is nxn filled with '0'
    B=cell(n,m);  [B{:}]=deal(0);
    C=cell(m,n);  [C{:}]=deal(0);
    D=cell(m,m);  [D{:}]=deal(0);
    id=cell(n,1);  [id{:}]=deal(0);
    e=cell(m,1);  [e{:}]=deal(0);
    jd=cell(m,1);  [jd{:}]=deal(0);
    v=compose('v_%d',(1:n)');          % v is filled with node names
    
    % We need to keep track of the number of voltage sources we've parsed
    vsCnt = 0; %not used
    
    % This loop does the bulk of filling in the arrays.  It scans line by line
    % and fills in the arrays depending on the type of element found on the
    % current line.

    for k1=1:nLines
        n1 = N1(k1);   % Get the two node numbers
        n2 = N2(k1);
        
        switch Name{k1}(1)
            % Passive element
            case {'R'} % RXXX N1 N2 VALUE

                % COULD INSERT CAPACITORS OR INDUCTORS IN FUTURE
                switch Name{k1}(1)  % Find 1/impedance for each element type.
                    case 'R'
                        g = 1/Value{k1};

                        any_nan = any(isnan(g), 'all');
                        any_inf = any(isinf(g), 'all');

                        %check if this becomes problematic
                        if any_nan || any_inf
                            fprintf('Element in g NaN or Inf, Value = %f\n',Value{k1})
                            
                        end 

                       
                end
                % Here we fill in G array by adding conductance.
                % The procedure is slightly different if one of the nodes is
                % ground, so check for thos accordingly.

                %GROUNDED CONTRIBUTES TO ONE ENTRY
                if (n1==0)
                    G{n2,n2} = G{n2,n2} + g;  % Add conductance.
                elseif (n2==0)
                    G{n1,n1} = G{n1,n1} + g;  % Add conductance.
                else
                    G{n1,n1} = G{n1,n1} + g;  % Add conductance.
                    G{n2,n2} = G{n2,n2} + g;  % Add conductance.
                    G{n1,n2} = G{n1,n2} - g;  % Sub conductance.
                    G{n2,n1} = G{n2,n1} - g;  % Sub conductance.
                end
                
            % Independent current source
            case 'I' % IXX N1 N2 VALUE  (Current N1 to N2)

                % Add current to nodes (if node is not ground)
                if n1~=0
                    id{n1} = (id{n1}) - Value{k1}; % subtract current from n1
                end
                if n2~=0
                    id{n2} = (id{n2}) + Value{k1}; % add current to n2
                end
        end
    end
    
    G_matrix = cell2mat(G);
    %check if elements along the diagonal are valid
    any_negative = any(diag(G_matrix) < 0);
    any_zero = any(diag(G_matrix) == 0);

    if any_negative
        disp('negative values found')
    elseif any_zero
        disp('zero value found')
    else
        disp('diagonal check passed')
    end
    
    %%  The submatrices are now complete.  Form the A, x, and z matrices,
    % and solve!
    
    A = ([G B; C D]); %Create and display A matrix
    
    %x = str2sym([v;jd])     %Create and display x matrix
    
    z = ([id;e])  ;     %Create and display z matrix

    A = cell2mat(A);
    z = cell2mat(z);
    %disp(A*x==z)
    
    A_pinv = pinv(A);

    a = pinv(A)*z;
    x = a;
    %a = A\z;  % Get the solution, this is where NaN first appears
        
%     for num=1:length(a)  % Assign each solution to its output variable.
%         eval(sprintf('%s = %s;',x(num),a(num)));
%     end
    
    %fprintf('\nThe solution:  \n');
    %x = eval(x);
    
    %display("iteration complete")
    %display(epsilons)
end








