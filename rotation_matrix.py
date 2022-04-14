import numpy as np

#conversion matrix between ra,dec and l,b

a_g = np.array([[-0.0548755604, +0.4941094279, -0.8676661490],
                [-0.8734370902, -0.4448296300, -0.1980763734], 
                [-0.4838350155, 0.7469822445, +0.4559837762]])


def Mrot(alpha_pole,delta_pole,phi1_0):
    '''
    Computes the rotation matrix to coordinates aligned with a pole 
    where alpha_pole, delta_pole are the poles in the original coorindates
    and phi1_0 is the zero point of the azimuthal angle, phi_1, in the new coordinates

    Critical: All angles must be in degrees!
    '''
    
    alpha_pole *= np.pi/180.
    delta_pole = (90.-delta_pole)*np.pi/180.
    phi1_0 *= np.pi/180.
    
    M1 = np.array([[np.cos(alpha_pole),np.sin(alpha_pole),0.],
                   [-np.sin(alpha_pole),np.cos(alpha_pole),0.],
                   [0.,0.,1.]])
    
    M2 = np.array([[np.cos(delta_pole),0.,-np.sin(delta_pole)],
                   [0.,1.,0.],
                   [np.sin(delta_pole),0.,np.cos(delta_pole)]])

    
    M3 = np.array([[np.cos(phi1_0),np.sin(phi1_0),0.],
                   [-np.sin(phi1_0),np.cos(phi1_0),0.],
                   [0.,0.,1.]])

    return np.dot(M3,np.dot(M2,M1))

def phi12(alpha,delta,alpha_pole,delta_pole,phi1_0):
    '''
    Converts coordinates (alpha,delta) to ones aligned with the pole (alpha_pole,delta_pole,phi1_0)
    
    Critical: All angles must be in degrees
    '''
    
    alpha = np.asarray(alpha)
    delta = np.asarray(delta)
    scalar_input = False

    if alpha.ndim == 0.:
        alpha = alpha[None]
        delta = delta[None]
        scalar_input = True
    
    vec_radec = np.array([np.cos(alpha*np.pi/180.)*np.cos(delta*np.pi/180.),np.sin(alpha*np.pi/180.)*np.cos(delta*np.pi/180.),np.sin(delta*np.pi/180.)])

    vec_phi12 = np.zeros(np.shape(vec_radec))

    R_phi12_radec = Mrot(alpha_pole,delta_pole,phi1_0)

    vec_phi12[0] = sum(R_phi12_radec[0][i]*vec_radec[i] for i in range(3))
    vec_phi12[1] = sum(R_phi12_radec[1][i]*vec_radec[i] for i in range(3))
    vec_phi12[2] = sum(R_phi12_radec[2][i]*vec_radec[i] for i in range(3))

    vec_phi12 = vec_phi12.T

    phi1 = np.arctan2(vec_phi12[:,1],vec_phi12[:,0])*180./np.pi
    phi2 = np.arcsin(vec_phi12[:,2])*180./np.pi


    if scalar_input:
        return [np.squeeze(phi1),np.squeeze(phi2)]
    else:
        return [phi1,phi2]


def phi12_rotmat(alpha,delta,R_phi12_radec):
    '''
    Converts coordinates (alpha,delta) to ones defined by a rotation matrix R_phi12_radec, applied on the original coordinates

    Critical: All angles must be in degrees
    '''

    alpha = np.asarray(alpha)
    delta = np.asarray(delta)
    scalar_input = False

    if alpha.ndim == 0.:
        alpha = alpha[None]
        delta = delta[None]
        scalar_input = True

    vec_radec = np.array([np.cos(alpha*np.pi/180.)*np.cos(delta*np.pi/180.),np.sin(alpha*np.pi/180.)*np.cos(delta*np.pi/180.),np.sin(delta*np.pi/180.)])

    vec_phi12 = np.zeros(np.shape(vec_radec))
    
    vec_phi12[0] = sum(R_phi12_radec[0][i]*vec_radec[i] for i in range(3))
    vec_phi12[1] = sum(R_phi12_radec[1][i]*vec_radec[i] for i in range(3))
    vec_phi12[2] = sum(R_phi12_radec[2][i]*vec_radec[i] for i in range(3))
    
    vec_phi12 = vec_phi12.T

    phi1 = np.arctan2(vec_phi12[:,1],vec_phi12[:,0])*180./np.pi
    phi2 = np.arcsin(vec_phi12[:,2])*180./np.pi


    if scalar_input:
        return [np.squeeze(phi1),np.squeeze(phi2)]
    else:
        return [phi1,phi2]

def pmphi12(alpha,delta,mu_alpha_cos_delta,mu_delta,R_phi12_radec):
    '''
    Converts proper motions (mu_alpha_cos_delta,mu_delta) to those in coordinates defined by the rotation matrix, R_phi12_radec, applied to the original coordinates

    Critical: All angles must be in degrees
    '''
    
    k_mu = 4.74047

    alpha = np.asarray(alpha)
    delta = np.asarray(delta)
    mu_alpha_cos_delta = np.asarray(mu_alpha_cos_delta)
    mu_delta = np.asarray(mu_delta)
    scalar_input = False
    
    if alpha.ndim == 0.:
        alpha = alpha[None]
        delta = delta[None]
        mu_alpha_cos_delta = mu_alpha_cos_delta[None]
        mu_delta = mu_delta[None]
        scalar_input = True
    

    
    phi1,phi2 = phi12_rotmat(alpha,delta,R_phi12_radec)


    r = np.ones(len(alpha))

    vec_v_radec = np.array([np.zeros(len(alpha)),k_mu*mu_alpha_cos_delta*r,k_mu*mu_delta*r]).T

    worker = np.zeros((len(alpha),3))

    worker[:,0] = ( np.cos(alpha*np.pi/180.)*np.cos(delta*np.pi/180.)*vec_v_radec[:,0]
                   -np.sin(alpha*np.pi/180.)*vec_v_radec[:,1]
                   -np.cos(alpha*np.pi/180.)*np.sin(delta*np.pi/180.)*vec_v_radec[:,2] )

    worker[:,1] = ( np.sin(alpha*np.pi/180.)*np.cos(delta*np.pi/180.)*vec_v_radec[:,0]
                   +np.cos(alpha*np.pi/180.)*vec_v_radec[:,1]
                   -np.sin(alpha*np.pi/180.)*np.sin(delta*np.pi/180.)*vec_v_radec[:,2] )

    worker[:,2] = ( np.sin(delta*np.pi/180.)*vec_v_radec[:,0]
                   +np.cos(delta*np.pi/180.)*vec_v_radec[:,2] )

    worker2 = np.zeros((len(alpha),3))

    worker2[:,0] = sum(R_phi12_radec[0][axis]*worker[:,axis] for axis in range(3))
    worker2[:,1] = sum(R_phi12_radec[1][axis]*worker[:,axis] for axis in range(3))
    worker2[:,2] = sum(R_phi12_radec[2][axis]*worker[:,axis] for axis in range(3))

    worker[:,0] = ( np.cos(phi1*np.pi/180.)*np.cos(phi2*np.pi/180.)*worker2[:,0]
                   +np.sin(phi1*np.pi/180.)*np.cos(phi2*np.pi/180.)*worker2[:,1]
                   +np.sin(phi2*np.pi/180.)*worker2[:,2] )

    worker[:,1] = (-np.sin(phi1*np.pi/180.)*worker2[:,0]
                   +np.cos(phi1*np.pi/180.)*worker2[:,1] )
                   

    worker[:,2] = (-np.cos(phi1*np.pi/180.)*np.sin(phi2*np.pi/180.)*worker2[:,0]
                   -np.sin(phi1*np.pi/180.)*np.sin(phi2*np.pi/180.)*worker2[:,1]
                   +np.cos(phi2*np.pi/180.)*worker2[:,2] )

    mu_phi1_cos_phi2 = worker[:,1]/(k_mu*r)
    mu_phi2 = worker[:,2]/(k_mu*r)

    if scalar_input:
        return [np.squeeze(mu_phi1_cos_phi2),np.squeeze(mu_phi2)]
    else:
        return [mu_phi1_cos_phi2,mu_phi2]

def pmphi12_reflex(alpha,delta,mu_alpha_cos_delta,mu_delta,R_phi12_radec,dist,vlsr=np.array([11.1,245,7.3])):
    
    ''' 
    returns proper motions in coordinates defined by R_phi12_radec transformation corrected by the Sun's reflex motion
    all angles must be in degrees, proper motions in mas/yr, and distance in kpc
    vlsr = np.array([11.1,244.,7.3])
    '''
    
    alpha = np.asarray(alpha)
    delta = np.asarray(delta)
    mu_alpha_cos_delta = np.asarray(mu_alpha_cos_delta)
    mu_delta = np.asarray(mu_delta)
    dist = np.asarray(dist)
    scalar_input = False
    
    if alpha.ndim == 0.:
        alpha = alpha[None]
        delta = delta[None]
        mu_alpha_cos_delta = mu_alpha_cos_delta[None]
        mu_delta = mu_delta[None]
        dist = dist[None]
        scalar_input = True

    k_mu = 4.74047

    a_g = np.array([[-0.0548755604, +0.4941094279, -0.8676661490],
                    [-0.8734370902, -0.4448296300, -0.1980763734], 
                    [-0.4838350155, 0.7469822445, +0.4559837762]])

    nvlsr = -vlsr

    phi1, phi2 = phi12_rotmat(alpha,delta,R_phi12_radec)

    phi1 = phi1*np.pi/180.
    phi2 = phi2*np.pi/180.

    pmphi1, pmphi2 = pmphi12(alpha,delta,mu_alpha_cos_delta,mu_delta,R_phi12_radec)

    M_UVW_phi12 = np.array([[np.cos(phi1)*np.cos(phi2),-np.sin(phi1),-np.cos(phi1)*np.sin(phi2)],
                            [np.sin(phi1)*np.cos(phi2), np.cos(phi1),-np.sin(phi1)*np.sin(phi2)],
                            [     np.sin(phi2)        ,      0.     , np.cos(phi2)]])

    vec_nvlsr_phi12 = np.dot(M_UVW_phi12.T,np.dot(R_phi12_radec,np.dot(a_g,nvlsr)))

    if scalar_input:
        return [np.squeeze(pmphi1 - vec_nvlsr_phi12[1]/(k_mu*dist)),np.squeeze(pmphi2 - vec_nvlsr_phi12[2]/(k_mu*dist))]
    else:
        return [pmphi1 - vec_nvlsr_phi12[1]/(k_mu*dist), pmphi2 - vec_nvlsr_phi12[2]/(k_mu*dist)]



def obs_from_pos6d(pos,vel,R_phi12_radec,R0=8.178,vlsr=np.array([11.1,245,7.3]),reflex_correction=False):

    '''
    returns observables (phi1,phi2,distance,pm1,pm2,vr) from a position and velocity vector
    pos is assumed to be in kpc and vel is assumed to be in km/s
    R_phi12_radec is the rotation matrix from ra,dec to phi12
    R0 is the Sun's distance from the galactic center in kpc
    vlsr is the Sun's velocity relative to the galaxy in km/s
    reflex_correction is a boolean which selects if the observables are reflex corrected
    '''

    R_phi12_lb = np.dot(R_phi12_radec,a_g)
    
    scalar_input = False
    if pos.ndim == 1:
        pos = pos[None]
        vel = vel[None]
        scalar_input = True
    
    pos = pos - np.array([-R0,0.,0.])
    
    theta = np.arctan2(pos[:,1],pos[:,0])
    theta = np.mod(theta,2.*np.pi)
    theta[theta > np.pi] -= 2.*np.pi
    phi = np.arcsin(pos[:,2]/(pos[:,0]**2.+pos[:,1]**2.+pos[:,2]**2.)**0.5)

    l = 180./np.pi*theta
    b = 180./np.pi*phi

    vec_lb = np.array([np.cos(l*np.pi/180.)*np.cos(b*np.pi/180.),np.sin(l*np.pi/180.)*np.cos(b*np.pi/180.),np.sin(b*np.pi/180.)])

    vec_phi12 = np.zeros(np.shape(vec_lb))

    vec_phi12[0] = sum(R_phi12_lb[0][i]*vec_lb[i] for i in range(3))
    vec_phi12[1] = sum(R_phi12_lb[1][i]*vec_lb[i] for i in range(3))
    vec_phi12[2] = sum(R_phi12_lb[2][i]*vec_lb[i] for i in range(3))

    vec_phi12 = vec_phi12.T

    phi1 = np.arctan2(vec_phi12[:,1],vec_phi12[:,0])*180./np.pi
    phi2 = np.arcsin(vec_phi12[:,2]/sum(vec_phi12[:,i]**2. for i in range(3))**0.5)*180./np.pi
    
    r_stream = sum(pos[:,i]**2. for i in range(3))**0.5
    
    if reflex_correction==False:
        vel = vel - vlsr

    vr_stream = sum((np.cos(phi1*np.pi/180.)*np.cos(phi2*np.pi/180.)*R_phi12_lb[0,axis]+np.sin(phi1*np.pi/180.)*np.cos(phi2*np.pi/180.)*R_phi12_lb[1,axis]+np.sin(phi2*np.pi/180.)*R_phi12_lb[2,axis])*vel[:,axis] for axis in range(3))
    mu_phi1_cos_phi2_stream = 1./(k_mu*r_stream)*sum( (-np.sin(phi1*np.pi/180.)*R_phi12_lb[0,axis]+np.cos(phi1*np.pi/180.)*R_phi12_lb[1,axis])*vel[:,axis] for axis in range(3))
    mu_phi2_stream = 1./(k_mu*r_stream)*sum( (-np.cos(phi1*np.pi/180.)*np.sin(phi2*np.pi/180.)*R_phi12_lb[0,axis] - np.sin(phi1*np.pi/180.)*np.sin(phi2*np.pi/180.)*R_phi12_lb[1,axis] + np.cos(phi2*np.pi/180.)*R_phi12_lb[2,axis])*vel[:,axis] for axis in range(3)) 

    #nvlsr = -vlsr
    #mu_phi1_cos_phi2_stream_corr =  mu_phi1_cos_phi2_stream - 1./(k_mu*r_stream)*sum( (-np.sin(phi1*np.pi/180.)*R_phi12_lb[0,axis]+np.cos(phi1*np.pi/180.)*R_phi12_lb[1,axis])*nvlsr[axis] for axis in range(3))
    #mu_phi2_stream_corr = mu_phi2_stream - 1./(k_mu*r_stream)*sum( (-np.cos(phi1*np.pi/180.)*np.sin(phi2*np.pi/180.)*R_phi12_lb[0,axis] - np.sin(phi1*np.pi/180.)*np.sin(phi2*np.pi/180.)*R_phi12_lb[1,axis] + np.cos(phi2*np.pi/180.)*R_phi12_lb[2,axis])*nvlsr[axis] for axis in range(3)) 
    #vr_stream_corr = vr_stream - sum( (np.cos(phi1*np.pi/180.)*np.cos(phi2*np.pi/180.)*R_phi12_lb[0,axis] + np.sin(phi1*np.pi/180.)*np.cos(phi2*np.pi/180.)*R_phi12_lb[1,axis] + np.sin(phi2*np.pi/180.)*R_phi12_lb[2,axis])*nvlsr[axis] for axis in range(3))

    
    if scalar_input:
        return np.squeeze(phi1),np.squeeze(phi2),np.squeeze(r_stream),np.squeeze(mu_phi1_cos_phi2_stream),np.squeeze(mu_phi2_stream),np.squeeze(vr_stream)
    else:
        return phi1,phi2,r_stream,mu_phi1_cos_phi2_stream,mu_phi2_stream,vr_stream

    
def pos6d_from_obs(phi1,phi2,dist,pm1,pm2,vr,R_phi12_radec,R0=8.178,vlsr=np.array([11.1,245,7.3]),reflex_correction=False):
    
    '''
    returns the position and velocity from a given set of observables in a coordinate system defined by
    the matrix R_phi12_lb (which takes a vector in lb to phi12)

    inputs:
    phi1,phi2 are assumed to be in degrees
    dist is in kpc
    pm1,pm2 are in mas/yr
    vr is in km/s
    R_phi12_radec is a rotation matrix to go from a vector in ra,dec to phi1,phi2

    outputs:
    pos is in kpc
    vel is in km/s
    '''

    R_phi12_lb = np.dot(R_phi12_radec,a_g)
    
    phi1 = np.asarray(phi1)
    phi2 = np.asarray(phi2)
    dist = np.asarray(dist)
    pm1  = np.asarray(pm1)
    pm2  = np.asarray(pm2)
    vr   = np.asarray(vr)
    scalar_input = False
    
    if phi1.ndim == 0:
        phi1 = phi1[None]
        phi2 = phi2[None]
        dist = dist[None]
        pm1  = pm1[None]
        pm2  = pm2[None]
        vr   = vr[None]
        scalar_input = True
    
    l,b = phi12_rotmat(phi1,phi2,R_phi12_lb.T)
    
    pos = np.zeros((len(phi1),3))
    
    pos[:,0] = -R0 + dist*np.cos(l*np.pi/180.)*np.cos(b*np.pi/180.)
    pos[:,1] =       dist*np.sin(l*np.pi/180.)*np.cos(b*np.pi/180.)
    pos[:,2] =       dist*np.sin(b*np.pi/180.)
    
    worker = np.zeros((len(phi1),3))
    
    phi1 = phi1*np.pi/180.
    phi2 = phi2*np.pi/180.
    
    worker[:,0] = vr*np.cos(phi1)*np.cos(phi2)-k_mu*dist*pm1*np.sin(phi1)-k_mu*dist*pm2*np.cos(phi1)*np.sin(phi2)
    worker[:,1] = vr*np.sin(phi1)*np.cos(phi2)+k_mu*dist*pm1*np.cos(phi1)-k_mu*dist*pm2*np.sin(phi1)*np.sin(phi2)
    worker[:,2] = vr*np.sin(phi2)+k_mu*dist*pm2*np.cos(phi2)
    
    vel = np.zeros((len(phi1),3))
    
    vel[:,0] = sum(R_phi12_lb[axis][0]*worker[:,axis] for axis in range(3))
    vel[:,1] = sum(R_phi12_lb[axis][1]*worker[:,axis] for axis in range(3))
    vel[:,2] = sum(R_phi12_lb[axis][2]*worker[:,axis] for axis in range(3))
    
    if reflex_correction == False:
        vel = vel + vlsr
    
    if scalar_input:    
        return np.squeeze(pos),np.squeeze(vel)
    else:
        return pos,vel
