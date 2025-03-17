## Dependences
import torch
import os
from absl import flags, app
import os
from train_calibrate import train
from test import test
from glbSettings import *



## Config Settings
# general
flags.DEFINE_enum('action', 'TRAIN', ['TRAIN','TEST'], 'Action: TRAIN or TEST.')
flags.DEFINE_string('basedir', './logs/', 'Where to store ckpts and logs.')
flags.DEFINE_string('expname', 'exp', help='Experiment name.')
flags.DEFINE_string('datadir', './data/LF/exp', 'Input data path.')
# embedder configs
flags.DEFINE_enum('embeddertype', 'None', ['None','PositionalEncoder'], 'Encoder type.')
# PositionalEncoder
flags.DEFINE_integer('multires', 10, 'log2 of max freq for positional encoding.')
flags.DEFINE_integer('multiresZ', None, 'log2 of max freq for positional encoding on z axis for anisotropic encoding.')
# model configs
flags.DEFINE_enum('modeltype', 'NeRF', ['NeRF','Grid','Grid_NeRF'], 'Model type.')
flags.DEFINE_boolean('no_reload', False, 'Do not reload weights from saved ckpt.')
flags.DEFINE_string('weights_path', None, 'Weights to be loaded from.')
flags.DEFINE_integer('sigch', 1, '#channels of the signal to be predicted.')
# NeRF-like
flags.DEFINE_integer('netdepth', 8, '#layers.')
flags.DEFINE_integer('netwidth', 256, '#channels per layer.')
flags.DEFINE_list('skips', [4], 'Skip connection layer indice.')
# Grid-INR
flags.DEFINE_string('grid_shape', None, 'Grid shape if use Grid_NeRF, should be in format "W H D".')
flags.DEFINE_float('overlapping', 0.0, 'Grid overlapping ratio. 0 for no overlap.')
# postprocessor configs
flags.DEFINE_enum('postprocessortype', 'relu', ['linear','relu','leak_relu'], 'Post processor type.')
# data options
flags.DEFINE_enum('datatype', 'LF', ['LF'], 'Dataset type: LF.')
# training options
flags.DEFINE_integer('N_steps', 10000, 'Number of training steps.')
flags.DEFINE_float('lrate', 5e-4, 'Learning rate.')
flags.DEFINE_float('lrate_kenel', 1e-2, 'Learning rate.')
flags.DEFINE_integer('lrate_decay', 250, 'Exponential learning rate decay (in 1000 steps).')
flags.DEFINE_integer('block_size', 129, 'Block Size trained one time. Should be an odd number.')
flags.DEFINE_float('L1_regu_weight', 0.1, 'Weight of l1 regularization term.')
flags.DEFINE_float('TV_regu_weight', 0.01, 'Weight of gradients regularization term.')
flags.DEFINE_float('ssim_ratio', 0.1, 'Weight of ssim loss.')
# rendering options
flags.DEFINE_integer('chunk', 64*1024, 'Number of pts sent through network in parallel, decrease if running out of memory.')
flags.DEFINE_list('render_size', None, 'Size of the extracted volume, should be in format "W,H,D". "None" for extract autometically from training data and H_mtx.')
flags.DEFINE_boolean('render_proj', False, 'Render out projections or not.')
flags.DEFINE_string('Hdir', None, 'H_mtx path for rendering projections. Omit when render_proj is set to False. None for reproducing training inputs.')
# logging/
flags.DEFINE_integer('i_print', 1000, 'Printout and logging interval.')
flags.DEFINE_boolean('shot_when_print', False, 'Save snapshot when logging.')
flags.DEFINE_float('psf_dz',4,'z sampling_rate of psf') # 0.1 * args.sampling_z 0.2
flags.DEFINE_float('psf_dy',1.8,'y sampling_rate of psf')
flags.DEFINE_float('psf_dx',1.8,'y sampling_rate of psf')#0.086
flags.DEFINE_float('n_detection', 0.13,'NA') #1.1
flags.DEFINE_float('emission_wavelength', 1,'wavelength')#0.515
flags.DEFINE_float('n_obj', 1.33,'refraction')
flags.DEFINE_float('zernike_max_val', 1e-2,'')
flags.DEFINE_integer('zernike_terms_num',4,'')
flags.DEFINE_integer('Nnum',9,'')
flags.DEFINE_boolean('aberration_correct', True, '')




FLAGS=flags.FLAGS

def main(argv):
    del argv
    if FLAGS.action.lower() == 'train':
        print(f'training  with aberration correction:{FLAGS.aberration_correct}')
        train(FLAGS)
    elif FLAGS.action.lower() == 'test':
        test(FLAGS)
    else:
        print("Unrecognized action. Do nothing.")



if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # np.random.seed(0)
    print(os.getcwd())
    if DEVICE.type == 'cuda':
        print(f"Run on CUDA:{GPU_IDX}")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("Run on CPU")
    torch.set_default_tensor_type('torch.FloatTensor')

    app.run(main)