%this script reads the voltage and power data for the MATLAB app to access


data200 = readmatrix('power_data200.csv');
data300 = readmatrix('power_data300.csv');
data400 = readmatrix('power_data400.csv');
data500 = readmatrix('power_data500.csv');
data600 = readmatrix('power_data600.csv');
data700 = readmatrix('power_data700.csv');
data800 = readmatrix('power_data800.csv');

v_data200 = readmatrix('voltage_data200.csv');
v_data300 = readmatrix('voltage_data300.csv');
v_data400 = readmatrix('voltage_data400.csv');
v_data500 = readmatrix('voltage_data500.csv');
v_data600 = readmatrix('voltage_data600.csv');
v_data700 = readmatrix('voltage_data700.csv');
v_data800 = readmatrix('voltage_data800.csv');

% %we want the positions of the defects
% defect1_tape = data200(:,1);
% defect1_slice = data200(:,2);
% defect2_tape = data200(:,3);
% defect2_slice = data200(:,4);
% 
% total_power = data200(:,5);
% 
% % Scatter plot with marker color representing total power
% figure()
% scatter(defect1_tape, defect1_slice, 50, total_power, 'filled');
% hold on;
% scatter(defect2_tape, defect2_slice, 50, total_power, 'filled');
% colorbar; % Add colorbar to show total power scale
% xlabel('X Position');
% ylabel('Y Position');
% title('Defect Positions with Total Power');