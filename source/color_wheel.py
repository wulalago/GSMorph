import numpy as np 
import matplotlib.pyplot as plt

def draw_quiver_fields(flow, ax, scale_down:int=2):
    '''
    flow -- deform fields with shape 2 H W
    type: Numpy array
    scale_factore defaut = 0.5
    '''
    # * Down sample the flow by the scale_factor 
    # flow = F.interpolate(input=flow,scale_factor=scale_factor,mode='bilinear')
    flow = flow[:,::scale_down,::scale_down]
    # *Get Shape of flow 
    C,H,W = flow.shape

    x,y = np.meshgrid(np.linspace(0,W-1,W),np.linspace(H-1,0,H))
    tmp_u = flow[0,...]
    tmp_v = flow[1,...]
    ax.quiver(x,y,tmp_u,tmp_v)
    ax.axis('off')
    ax.set_aspect('equal')

def save_fig(img,save=False,dir:str='./fig.png'):
    fig, ax = plt.subplots() 
    ax.imshow(img)
    # plot_grid(grid_c+flow[1,::down_sample,::down_sample], grid_r+flow[0,::down_sample,::down_sample], ax=ax, color="red")
    plt.axis('off')
    if save:
        plt.savefig(dir, bbox_inches='tight',pad_inches = 0, dpi=300)
        plt.close(fig)

    else:
        plt.show()

from matplotlib.collections import LineCollection

def plot_grid(x,y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, linewidths=0.15, **kwargs))
    ax.add_collection(LineCollection(segs2, linewidths=0.15, **kwargs))
    ax.autoscale()
    
def draw_dvf(flow, ax, down_sample:int=1):
    # *INPUT: c, H, W -> flow numpy 
    c,h,w = flow.shape 
    # *form grid 
    grid_c,grid_r = np.meshgrid(np.linspace(0,w-1,w//down_sample),np.linspace(0,h-1,h//down_sample))
    flow_img = flow2img(flow)
    ax.imshow(flow_img)
    plot_grid(grid_c+flow[1,::down_sample,::down_sample], grid_r+flow[0,::down_sample,::down_sample], ax=ax, color="red")
    plt.axis('off')
    
def ill_dvf(flow,save=False,dir:str='./dvfs.png',down_sample:int=1):
    # *INPUT: c, H, W -> flow numpy 
    c,h,w = flow.shape 
    # *form grid 
    grid_c,grid_r = np.meshgrid(np.linspace(0,w-1,w//down_sample),np.linspace(0,h-1,h//down_sample))
    flow_img = flow2img(flow)
    fig, ax = plt.subplots() 
    ax.imshow(flow_img)
    plot_grid(grid_c+flow[1,::down_sample,::down_sample], grid_r+flow[0,::down_sample,::down_sample], ax=ax, color="red")
    plt.axis('off')
    if save:
        plt.savefig(dir, bbox_inches='tight',pad_inches = 0, dpi=300)
        plt.close(fig)

    else:
        plt.show()

def flow2img(flow_data):
    '''
    convert optical flow into color image
    :param flow_data: Numpy 2,H,W
    :return: color image
    '''
    u = flow_data[0,...]
    v = flow_data[1,...]

    UNKNOW_FLOW_THRESHOLD = 1e7
    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    # get max value in each direction
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)
    # print(img)

    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
	"""
	compute optical flow color map
	:param u: horizontal optical flow
	:param v: vertical optical flow
	:return:
	"""

	height, width = u.shape
	img = np.zeros((height, width, 3))

	NAN_idx = np.isnan(u) | np.isnan(v)
	u[NAN_idx] = v[NAN_idx] = 0

	colorwheel = make_color_wheel()
	ncols = np.size(colorwheel, 0)

	rad = np.sqrt(u ** 2 + v ** 2)

	a = np.arctan2(-v, -u) / np.pi

	fk = (a + 1) / 2 * (ncols - 1) + 1

	k0 = np.floor(fk).astype(int)

	k1 = k0 + 1
	k1[k1 == ncols + 1] = 1
	f = fk - k0

	for i in range(0, np.size(colorwheel, 1)):
		tmp = colorwheel[:, i]
		col0 = tmp[k0 - 1] / 255
		col1 = tmp[k1 - 1] / 255
		col = (1 - f) * col0 + f * col1

		idx = rad <= 1
		col[idx] = 1 - rad[idx] * (1 - col[idx])
		notidx = np.logical_not(idx)

		col[notidx] *= 0.75
		img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

	return img


def make_color_wheel():
	"""
	Generate color wheel according Middlebury color code
	:return: Color wheel
	"""
	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR

	colorwheel = np.zeros([ncols, 3])

	col = 0

	# RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
	col += RY

	# YG
	colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
	colorwheel[col:col + YG, 1] = 255
	col += YG

	# GC
	colorwheel[col:col + GC, 1] = 255
	colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
	col += GC

	# CB
	colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
	colorwheel[col:col + CB, 2] = 255
	col += CB

	# BM
	colorwheel[col:col + BM, 2] = 255
	colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
	col += + BM

	# MR
	colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
	colorwheel[col:col + MR, 0] = 255

	return colorwheel