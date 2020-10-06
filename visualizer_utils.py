import numpy as np
import copy

def generate_points(range_x, range_y, z, total, object_x_y_size):
    rx = copy.deepcopy(range_x)
    ry = copy.deepcopy(range_y)
    rx[0] += object_x_y_size[0]
    rx[1] -= object_x_y_size[0]
    ry[0] += object_x_y_size[1]
    ry[1] -= object_x_y_size[1]
    xs = np.linspace(start=rx[0], stop=rx[1], num=total, endpoint=True)
    ys = np.linspace(start=ry[0], stop=ry[1], num=total, endpoint=True)
    points = []
    for i in range(total):
        for j in range(total):
            points.append([xs[i], ys[j], z])
    return points

# todo set to false once is trained with table
def take_obstacle_image(env, img_size, make_table_invisible=True):
    env.env._set_arm_visible(visible=False)
    env.env._set_visibility(names_list=['obstacle'], alpha_val=1.0)
    env.env._set_visibility(names_list=['object0'], alpha_val=0.0)
    if not make_table_invisible:
        env.env._set_visibility(names_list=['table0'], alpha_val=1.0)
    rgb_array = np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
    return rgb_array

def take_goal_image(env, img_size, make_table_invisible=True, make_walls_invisible=True):
    env.env._set_arm_visible(visible=False)
    env.env._set_visibility(names_list=['object0'], alpha_val=1.0)
    if not make_table_invisible:
        env.env._set_visibility(names_list=['table0'], alpha_val=1.0)
    if not make_walls_invisible:
        if 'wall1' in env.env.sim.model.body_names:
            env.env._set_visibility(names_list=['wall1', 'wall2', 'wall3', 'wall4'], alpha_val=1.0)
    rgb_array = np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
    return rgb_array

def take_env_image(env, img_size):
    env.env._set_arm_visible(visible=False)
    env.env._set_visibility(names_list=['object0'], alpha_val=1.0)
    env.env._set_visibility(names_list=['obstacle'], alpha_val=1.0)
    env.env._set_visibility(names_list=['table0'], alpha_val=1.0)
    for id in  [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16, 17, 18, 21]:
        env.env._set_visibility_with_id(id, alpha_val=0.2)
    #just to activate in case viewer is not intialized
    if not hasattr(env.env.viewer, 'cam'):
        np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
    rgb_array = np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
    return rgb_array

def take_objects_image(env, img_size):
    env.env._set_arm_visible(visible=False)
    env.env._set_visibility(names_list=['rectangle'], alpha_val=1.0)
    env.env._set_visibility(names_list=['cube'], alpha_val=1.0)
    env.env._set_visibility(names_list=['cylinder'], alpha_val=1.0)
    env.env._set_visibility(names_list=['table0'], alpha_val=0.0)
    # just to activate in case viewer is not intialized
    if not hasattr(env.env.viewer, 'cam'):
        np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
    rgb_array = np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
    return rgb_array