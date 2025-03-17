% %% Import dependecies
addpath ./util
import2ws();
clc;
clear all;
% Import configurations of FLFM setup
Config.gridType =  'Reg';
Config.focus_shift_ratio=0.3;
Config.NA =1;
Config.M=27.78;%27.78;
Config.f1 = 200000;
Config.fobj = Config.f1/Config.M;         %focal length of the objective lens. Camera.fobj = Camera.ftl/Camera.M

Config.f2 = 200000;%150000
Config.fm = 30000;          %20000、15000 %fm - focal length of the micro-lens.
Config.mla2sensor = Config.fm;   %distance between the MLA plane and camera sensor plane.

Config.lensPitch = 1700;%1000/1500
Config.d1=2300;%2400/18000
Config.WaveLength = 1010*1e-3;
Config.pixelPitch = 15;  %the number of pixels between two horizontally neighboring lenslets.
Config.spacingPixels = Config.d1/Config.pixelPitch;  % sensor pixel pitch (in ?m).
Config.MLAPixels = Config.lensPitch/Config.pixelPitch;  % sensor pixel pitch (in ?m).

% Config.lensPitch = Config.pixelPitch*Config.spacingPixels;

Config.immersion_n=1.33;
Config.n  = 1;
superResFactor =2; % superResFactor controls the lateral resolution of the reconstructed object. It is interpreted as a multiple of the lenslet resolution (1 voxel/lenslet). superResFactor =3default1 means the object is reconstructed at sensor resolution, while superResFactor = 1 means lenslet resolution.
Config.SensorSize = [512,640].*superResFactor; %the size of the input light field image,[512,640]

Config.view_num=9;
Config.coordi=zeros(Config.view_num,2);%coordi is the coordinate of center of each microlens on camera Sensor
Config.coordi(:,1)=[100, 108, 101, 252, 256,263 ,414, 413,409]'.*superResFactor+1;
Config.coordi(:,2)=[161, 313, 472, 174, 320,467, 158, 322,464]'.*superResFactor+1;

Config.X_center = ceil(Config.SensorSize(1)/2);
Config.Y_center = ceil(Config.SensorSize(2)/2);
Config.depthStep = 4;
% Config.depthRange = [0,0];
Config.depthRange =[-88 ,88];%[-112 ,112]、[-88 ,88] ;
Config.MLAnumX = 3;
Config.MLAnumY = 3;

d_max=floor(Config.d1*sqrt(2));
D_pupuil=Config.f2*Config.NA*2/Config.M;
Config.FOV=Config.d1*Config.f2/Config.fm/Config.M;
system_magnification=Config.f1/Config.fobj*Config.fm/Config.f2;
rxy=Config.WaveLength*Config.fm/Config.lensPitch/system_magnification;

rz=rxy*Config.f2/d_max/Config.M;

sr_=rxy/(Config.pixelPitch/system_magnification);
N_=rxy*2.*Config.NA/Config.WaveLength;

tan_theta=Config.lensPitch/Config.f2;
DOF_ge_ideal=Config.lensPitch*Config.f2/Config.fm/(tan_theta*Config.M^2);
Config.DOF_wave_ideal=2*(4+2/sr_)*rxy^2/Config.WaveLength;

fprintf('[!]System Magnification: %.4f Optical Resolution (Rxy):%.4f (Rz): %.4f Voxel_size:(%.3f,%.3f)\n',system_magnification,rxy,rz,Config.pixelPitch/system_magnification,Config.depthStep);
fprintf('[!]System fov:%.3f DOF:%.3f D_pupuil:%.1f',Config.FOV,Config.DOF_wave_ideal,D_pupuil);
fprintf('[!]System Nnum:%d mla_num (%d) Pitch size: %.2f\n',Config.spacingPixels,Config.view_num,Config.pixelPitch);



%%axicon
Config.n_axicon=1.534;
Config.theta=2*pi/180;
mla2axicon=0;

beta_=atan(Config.lensPitch/2/Config.fm);
h_=Config.lensPitch/2-mla2axicon*Config.lensPitch/2/Config.fm;
gamma=asin(  Config.n_axicon *sin( beta_+Config.theta  ))-Config.theta;
NA_gamma=1*sin(gamma);
Rxy_gamma=Config.WaveLength/2/NA_gamma/system_magnification;
z_focus=h_/tan(gamma);
z_inference_max=Config.lensPitch*Config.fm/(2*Config.fm*(Config.n_axicon-1)*Config.theta+Config.lensPitch)-mla2axicon;
[Camera,LensletGridModel] = Compute_camera(Config,superResFactor);
Resolution = Compute_Resolution(Camera,LensletGridModel);

is_axicon=0;
if ~is_axicon
    ax2ca=Config.fm;
%     ax2ca=Config.fm-Config.focus_shift_ratio*Config.depthRange(2)*system_magnification^2;
%     disp('multi_focus');
else
    ax2ca=18258776810.2/1e6;
%     ax2ca=z_inference_max-3000:500:z_inference_max;
end
[H, Ht] = FLFM_computeLFMatrixOperators(Camera, Resolution,ax2ca,mla2axicon,is_axicon);



