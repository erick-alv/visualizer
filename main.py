from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from j_vae.train_vae_sb import load_Vae as load_Vae_SB
from j_vae.train_vae import load_Vae
from visualizer_utils import generate_points, take_goal_image, take_obstacle_image, take_objects_image
import gym


range_x=[1.05, 1.55]
range_y=[0.5, 1.0]
z_table_height=0.43
vae_sb_weights_file_name = {'goal': 'vae_sb_model_goal',
                            'obstacle':'vae_sb_model_obstacle',
                            'mixed':'vae_sb_model_mixed',
                            'all':'all_sb_model'
                            }

def visualization_grid_points(env, model, size_to_use, img_size, n, enc_type, ind_1, ind_2, fig_file_name,
                              using_sb=True,):

    points = generate_points(range_x=range_x, range_y=range_y, z=z_table_height, total=n,
                             object_x_y_size=[size_to_use, size_to_use])
    n_labels = np.arange(len(points))
    points = np.array(points)

    xs = points[:, 0]
    ys = points[:, 1]
    plt.figure(1)
    plt.subplot(211, )
    plt.scatter(xs,ys)
    plt.title('real')
    for i, en in enumerate(n_labels):
        plt.annotate(en, (xs[i], ys[i]))


    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    # sample images
    data_set = np.empty([len(points), img_size, img_size, 3])
    #move other objects to placess they do not collide
    if enc_type == 'goal' or (args.enc_type == 'mixed' and args.mix_h == 'goal'):
        env.env._set_position(names_list=['obstacle'], position=[2., 2., 0.4])
    elif enc_type == 'obstacle' or (args.enc_type == 'mixed' and args.mix_h == 'obstacle'):
        env.env._move_object(position=[2.,2.,0.4])
    else:
        raise Exception('Not supported enc type')
    for i,p in enumerate(points):
        if enc_type == 'goal' or (args.enc_type == 'mixed' and args.mix_h == 'goal'):
            env.env._move_object(position=p)
            data_set[i] = take_goal_image(env, img_size, make_table_invisible=True)
        elif enc_type == 'obstacle' or (args.enc_type == 'mixed' and args.mix_h == 'obstacle'):
            env.env._set_position(names_list=['obstacle'], position=p)
            data_set[i] = take_obstacle_image(env, img_size)
        else:
            raise Exception('Not supported enc type')

    #images from the positions
    all_array = None
    t = 0
    for r in range(n):
        row = None
        for c in range(n):
            rcim = data_set[t].copy()
            t += 1
            if row is None:
                row = rcim
            else:
                row = np.concatenate([row.copy(), rcim], axis=1)
        if all_array is None:
            all_array = row.copy()
        else:
            all_array = np.concatenate([all_array.copy(), row], axis=0)
    all_ims = Image.fromarray(all_array.astype(np.uint8))
    all_ims.save('{}_ims.png'.format(fig_file_name))
    all_ims.close()

    data = torch.from_numpy(data_set).float().to(device)
    data /= 255
    data = data.permute([0, 3, 1, 2])
    model.eval()
    if not using_sb:
        mu, logvar = model.encode(data.reshape(-1, img_size * img_size * 3))
    else:
        mu, logvar = model.encode(data)
    mu = mu.detach().cpu().numpy()

    assert ind_1 != ind_2
    mu = np.concatenate([np.expand_dims(mu[:, ind_1], axis=1),
                         np.expand_dims(mu[:, ind_2], axis=1)], axis=1)


    if enc_type == 'goal' or (args.enc_type == 'mixed' and args.mix_h == 'goal'):
        #rm = create_rotation_matrix(angle_goal)
        #mu = rotate_list_of_points(mu, rm)
        #mu = map_points(mu, goal_map_x, goal_map_y)
        pass
    elif enc_type == 'obstacle' or (args.enc_type == 'mixed' and args.mix_h == 'obstacle'):
        #rm = create_rotation_matrix(angle_obstacle)
        #mu = rotate_list_of_points(mu, rm)
        #mu = map_points(mu, obstacle_map_x, obstacle_map_y)
        pass
    else:
        raise Exception('Not supported enc type')

    lxs = mu[:, 0]
    lys = mu[:, 1]
    plt.subplot(212)
    plt.scatter(lxs, lys)
    plt.title('latent')
    for i, en in enumerate(n_labels):
        plt.annotate(en, (lxs[i], lys[i]))

    plt.savefig(fig_file_name)
    plt.close()



if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='gym env id', type=str, default='FetchReach-v1', required=True)
    args, _ = parser.parse_known_args()

    parser.add_argument('--ind_1', help='first index to extract from latent vector', type=np.int32, default=0)
    parser.add_argument('--ind_2', help='second index to extract from latent vector', type=np.int32, default=1)

    parser.add_argument('--enc_type', help='the type of attribute that we want to generate/encode', type=str,
                        default='goal', choices=['goal', 'obstacle', 'obstacle_sizes', 'goal_sizes', 'mixed', 'all'])
    parser.add_argument('--mix_h', help='if the representation should de done with goals or obstacles', type=str,
                        default='goal', choices=['goal', 'obstacle'])

    parser.add_argument('--img_size', help='size image in pixels', type=np.int32, default=84)
    parser.add_argument('--latent_size', help='latent size to train the VAE', type=np.int32, default=2)


    args = parser.parse_args()

    # get names corresponding folders, and files where to store data
    this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
    base_data_dir = this_file_dir + 'data/'
    data_dir = base_data_dir + args.env + '/'
    weights_path = data_dir + vae_sb_weights_file_name[args.enc_type]


    # load environment
    env = gym.make(args.env)

    #other arguments for the algorithms

    #set to 0 so positioning start at the beginning
    size_to_use = 0


    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if args.enc_type == 'goal' or args.enc_type == 'obstacle' or args.enc_type == 'mixed':

        model = load_Vae_SB(weights_path, args.img_size, args.latent_size)
    elif args.enc_type == 'all':
        model = load_Vae_SB(weights_path, args.img_size, args.latent_size,
                            full_connected_size=640, extra_layer=True)
    else:
        model = load_Vae(weights_path, args.imgsize, args.latent_size)

    if args.enc_type == 'goal' or (args.enc_type == 'mixed' and args.mix_h == 'goal'):
        fig_name = 'vis_grid_g'
    elif args.enc_type == 'obstacle' or (args.enc_type == 'mixed' and args.mix_h == 'obstacle'):
        fig_name = 'vis_grid_o'

    visualization_grid_points(n=7, env=env, model=model,size_to_use=size_to_use, img_size=args.img_size,
                              enc_type=args.enc_type, ind_1=args.ind_1, ind_2=args.ind_2,
                              fig_file_name=fig_name)
