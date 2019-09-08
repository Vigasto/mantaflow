#
# MPM with APIC velocity update + adaptive timestepping
# TODO : MPM gradient part
#
from manta import *

# solver params
dim = 3
particleNumber = 2
res = 64
gs = vec3(res,res,res)
if (dim==2):
	gs.z=1
	particleNumber = 3      # use more particles in 2d

s = Solver(name='main', gridSize = gs, dim=dim)

# how many frames to calculate 
frames    = 2500

# adaptive time stepping
s.frameLength = 0.6   # length of one frame (in "world time")
s.timestepMin = 0.1   # time step range
s.timestepMax = 2.0
s.cfl         = 1.5   # maximal velocity per cell
s.timestep    = (s.timestepMax+s.timestepMin)*0.5

minParticles = pow(2, dim)
timings = Timings()

# size of particles
radiusFactor = 1.0

# prepare grids and particles
flags    = s.create(FlagGrid)
phi      = s.create(LevelsetGrid)

vel      = s.create(MACGrid)
velOld   = s.create(MACGrid)
pressure = s.create(RealGrid)
tmpVec3  = s.create(VecGrid)

mesh     = s.create(Mesh)
displayMesh = true

pp       = s.create(BasicParticleSystem)
# add velocity data to particles
pVel     = pp.create(PdataVec3)
# apic part
mass  = s.create(MACGrid)
pCx   = pp.create(PdataVec3)
pCy   = pp.create(PdataVec3)
pCz   = pp.create(PdataVec3)

# acceleration data for particle nbs
pindex = s.create(ParticleIndexSystem) 
gpi    = s.create(IntGrid)

# scene setup, 0=breaking dam, 1=drop into pool
setup = 1
bWidth=1
flags.initDomain(boundaryWidth=bWidth)
fluidVel = 0
fluidSetVel = 0

if setup==0:
	# breaking dam
	fluidbox = Box( parent=s, p0=gs*vec3(0,0,0), p1=gs*vec3(0.4,0.6,1)) # breaking dam
	#fluidbox = Box( parent=s, p0=gs*vec3(0.4,0.72,0.4), p1=gs*vec3(0.6,0.92,0.6)) # centered falling block
	phi = fluidbox.computeLevelset()
elif setup==1:
	# falling drop
	fluidBasin = Box( parent=s, p0=gs*vec3(0,0,0), p1=gs*vec3(1.0,0.1,1.0)) # basin
	dropCenter = vec3(0.5,0.3,0.5)
	dropRadius = 0.1
	fluidDrop  = Sphere( parent=s , center=gs*dropCenter, radius=res*dropRadius)
	fluidVel   = Sphere( parent=s , center=gs*dropCenter, radius=res*(dropRadius+0.05) )
	fluidSetVel= vec3(0,-0.03,0)
	phi = fluidBasin.computeLevelset()
	phi.join( fluidDrop.computeLevelset() )

flags.updateFromLevelset(phi)
#setOpenBound(flags,bWidth,'xX',FlagOutflow|FlagEmpty)
sampleLevelsetWithParticles(phi=phi, flags=flags, parts=pp, discretization=particleNumber, randomness=0.05)
mapGridToPartsVec3(source=vel, parts=pp, target=pVel )

if fluidVel!=0:
	# set initial velocity
	fluidVel.applyToGrid( grid=vel , value=gs*fluidSetVel )
	mapGridToPartsVec3(source=vel, parts=pp, target=pVel )

lastFrame = -1
if (GUI):
	gui = Gui()
	gui.show()
	#gui.pause()

    # show all particles shaded by velocity
	gui.nextPdata()
	gui.nextPartDisplay()
	gui.nextPartDisplay()

#main loop
while s.frame < frames:
	maxVel = vel.getMax()
	s.adaptTimestep( maxVel )
	mantaMsg('\nFrame %i, time-step size %f' % (s.frame, s.timestep))

	# APIC
	pp.advectInGrid(flags=flags, vel=vel, integrationMode=IntRK4, deleteInObstacle=False)
	
    # set velocities through liquid marked regions
	apicMapPartsToMAC(flags=flags, vel=vel, parts=pp, partVel=pVel, cpx=pCx, cpy=pCy, cpz=pCz, mass=mass)
	extrapolateMACFromWeight(vel=vel , distance=2, weight=tmpVec3)
	markFluidCells(parts=pp, flags=flags)

	# create approximate surface level set, resample particles
	gridParticleIndex( parts=pp , flags=flags, indexSys=pindex, index=gpi )
	unionParticleLevelset( pp, pindex, flags, gpi, phi , radiusFactor ) 
	resetOutflow(flags=flags,parts=pp,index=gpi,indexSys=pindex) 
	# extend levelset somewhat, needed by particle resampling in adjustNumber
	extrapolateLsSimple(phi=phi, distance=4, inside=True); 

    # forces & pressure solve
	addGravity(flags=flags, vel=vel, gravity=(0,-0.003,0))
	setWallBcs(flags=flags, vel=vel)
	solvePressure(flags=flags, vel=vel, pressure=pressure)
	setWallBcs(flags=flags, vel=vel)

	# set source grids for resampling, used in adjustNumber!
	pVel.setSource( vel, isMAC=True )
	adjustNumber( parts=pp, vel=vel, flags=flags, minParticles=1*minParticles, maxParticles=2*minParticles, phi=phi, radiusFactor=radiusFactor ) 

	# we dont have any levelset, ie no extrapolation, so make sure the velocities are valid
	extrapolateMACSimple(flags=flags, vel=vel)

	# APIC velocity update
	apicMapMACGridToParts(partVel=pVel, cpx=pCx, cpy=pCy, cpz=pCz, parts=pp, vel=vel, flags=flags)

	if displayMesh and (dim==3):
		phi.createMesh(mesh)

	#gui.screenshot('flipt_%04d.png' % t);
	s.step()

	lastFrame = s.frame
