#!/usr/local/bin/python
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import shapefile
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import seaborn as sns

#
# parameters:

zctaShapeLoc = 'G:/CISS-West/Value Institute/Users/Matthew/US_zcta_codes/2015/cb_2015_us_zcta510_500k'

# tools to do mapping, etc.

def zcta_geographies(zctaShapeLoc=zctaShapeLoc, goodcoords=[], zctaform='str'):
    '''
    Defines the geographies of zip code tabulation areas (zcta's)
    in the united states.

    goodcoords = a coordinate set [lowlong, lowlat, hilong, hilat]
    outside of which a zcta will not be kept
    can be left blank (then all will be taken)
    goodcoords=[-91.2, 40.8, -81.8, 48.2] is the square surrounding Michigan.

    '''
    sf = shapefile.Reader(zctaShapeLoc)

    if len(goodcoords)>0:
        LOlong = goodcoords[0]
        LOlat = goodcoords[1]
        HIlong = goodcoords[2]
        HIlat = goodcoords[3]

    # Find zipcode and the shape of each zipcode:
    zctacodes = []
    zctashapes = []
    allrecords = sf.shapeRecords()
    for record in allrecords:
        zctacode = record.record[0]
        zctashape = record.shape

        if len(goodcoords)>0:
            lolong = record.shape.bbox[0]
            lolat = record.shape.bbox[1]
            hilong = record.shape.bbox[2]
            hilat = record.shape.bbox[3]
            condition = lolong>LOlong and lolat>LOlat and hilong<HIlong and hilat<HIlat
            if condition==True:
                zctacodes.append(zctacode)
                zctashapes.append(zctashape)

        else:
            zctacodes.append(zctacode)
            zctashapes.append(zctashape)

    # modify output if needed:
    if zctaform == 'np.int':
        zctacodes = [np.int64(code) for code in zctacodes]

    return zctacodes, zctashapes


def distance_haversine(lon1, lat1, lon2, lat2):
    '''
    Haversine formula, to calc distance given 2 points'
    longitudes and latitudes

    from https://gist.github.com/rochacbruno/2883505
    '''

#    lat1, lon1 = origin
#    lat2, lon2 = destination
#    radius = 6371 # km
    radius = 3959 # miles
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d


def shape_centers(shapenames, shapefiles):
    # Get the (approx) centers of the shapes:
    centers_long = []
    centers_lat = []
    for shape in shapefiles:
        center_long = np.mean([shape.bbox[0], shape.bbox[2]])
        center_lat = np.mean([shape.bbox[1], shape.bbox[3]])

        centers_long.append(center_long)
        centers_lat.append(center_lat)

    # create df of the shape centers:
    df_shapecenters = pd.DataFrame({'longitude':centers_long,
                                  'latitude':centers_lat})
    df_shapecenters.index = shapenames

    return df_shapecenters


def build_geodist_matrix(shapenames, shapefiles, distmetric='haversine'):
    '''
    Build a distance matrix from the center of each shape
    to the center of each other shape in a list of shapes.

    Assumed that these are geographical shapes, with longitudes
    and latitudes.

    Ex:
    shapenames = list of unique identifiers for the shapes.
    shapefiles = list of shapefiles

    if starting with a shaperecord (ex below):
    sf = shapefile.Reader("/Users/matto/Documents/taxidata/ZillowNeighborhoods-NY/ZillowNeighborhoods-NY")
    allrecords = sf.shapeRecords()
    shapefiles = [shape.shape for shape in allrecords]

    '''

    # check that shapenames are unique:
    assert len(shapenames)==len(np.unique(shapenames)), 'shapenames must be unique'

    df_shapecenters = shape_centers(shapefiles)

    # build shape distances matrix:
    mat_geodists = np.zeros((len(shapenames),len(shapenames)))
    for n1 in range(len(shapenames)):
        if n1 % 50 == 0:
            print 'building row %s of %s' % (n1, len(shapenames))
        for n2 in range(len(shapenames)):
            shape1 = shapenames[n1]
            shape2 = shapenames[n2]
            lon1 = df_shapecenters.loc[shape1,'longitude']
            lon2 = df_shapecenters.loc[shape2, 'longitude']
            lat1 = df_shapecenters.loc[shape1,'latitude']
            lat2 = df_shapecenters.loc[shape2, 'latitude']

            if distmetric=='haversine':
                geodist = distance_haversine(lon1, lat1, lon2, lat2)
            elif distmetric=='simple':
                dlong = lon2 - lon1
                dlat = lat2 - lat1
                geodist = np.sqrt(dlong**2 + dlat**2)

            mat_geodists[n1, n2] = geodist

    return df_shapecenters, mat_geodists


def zctas_within_radius(Clat, Clong, radius, zctacodes, zctashapes, df_shapecenters=[], includeRule='inclusive'):
    '''
    lat = latitude at the center
    long = longitude at center
    radius = desired radius in miles
    includeRule = what to do with zctas on the border
    '''

    # create df_shapecenters if needed
    if len(df_shapecenters) == 0:
        df_shapecenters = shape_centers(zctacodes, zctashapes)

    # simple inclusion rule:
    included_zctas = []
    for zcta in df_shapecenters.index:
        Zlat = np.float64(df_shapecenters.loc[[zcta]]['latitude'])
        Zlong = np.float64(df_shapecenters.loc[[zcta]]['longitude'])

        dist = distance_haversine(Clong, Clat, Zlong, Zlat)

        if dist <= radius:
            included_zctas.append(zcta)

    return included_zctas


### experimental: trying to draw zip codes (see leadtools.py)

def draw_zips(zctashapes):

    # set style:
    sns.set(style="white", color_codes=True, font_scale=1.5)

    #   -- input --
    #recs = zctacodes
    shapes = zctashapes
    Nshp    = len(shapes)

#    cm    = plt.get_cmap('Dark2')
#    cccol = cm(1.*np.arange(Nshp)/Nshp)
#    cccol = cm(0*np.arange(Nshp))
    #   -- plot --
    fig     = plt.figure()
    ax      = fig.add_subplot(111)
    for nshp in xrange(Nshp):
        ptchs   = []
        pts     = np.array(shapes[nshp].points)
        prt     = shapes[nshp].parts
        par     = list(prt) + [pts.shape[0]]
        for pij in xrange(len(prt)):
            ptchs.append(Polygon(pts[par[pij]:par[pij+1]]))
#        ax.add_collection(PatchCollection(ptchs,facecolor=cccol[nshp,:],edgecolor='k', linewidths=.1))
            ax.add_collection(PatchCollection(ptchs,edgecolor='k', facecolor='w', linewidths=.1))
    ax.set_xlim(-91,-82)
    ax.set_ylim(41,48)

    # return to default (this is a hack..)
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    # how scale colors?
    # how add colorbar?

    return ax


def draw_zipcenters(zctacodes, zctashapes, dfl, colorcol='', gamma=1.0):
    '''
    Needs to be vetted..
    '''
    dfl_zctashapes, dfl_zctacenters, dfl_cities = \
        zctas_for_dfl(zctacodes, zctashapes, dfl)

    # set style:
    sns.set(style="white", color_codes=True, font_scale=1.5)
#    sns.color_palette("Blues")

    # plot:
#    plt.figure()

    # determine what column to color by:
    if len(colorcol)>0:
        c = dfl[colorcol].copy()#**gamma
        plt.scatter(dfl_zctacenters['longitude'],\
                    dfl_zctacenters['latitude'],c=c, norm=colors.PowerNorm(gamma=gamma), cmap='Reds')
    else:
        plt.scatter(dfl_zctacenters['longitude'],\
                    dfl_zctacenters['latitude'])

    plt.colorbar()
#    plt.show()

    # return to default (this is a hack..)
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    return dfl_zctashapes, dfl_zctacenters












#
# def draw_zips_2(zctashapes, dfl, colorcol='', gamma=1.0):

#     if len(colorcol)>0:
#         c = dfl[colorcol].copy()#**gamma


#     shapes = zctashapes
#     Nshp    = len(shapes)

#     cm    = plt.get_cmap('Dark2')
#     cccol = cm(1.*c/max(c))

#     #   -- plot --
#     fig     = plt.figure()
#     ax      = fig.add_subplot(111)
#     for nshp in xrange(Nshp):
#         ptchs   = []
#         pts     = np.array(shapes[nshp].points)
#         prt     = shapes[nshp].parts
#         par     = list(prt) + [pts.shape[0]]

#         for pij in xrange(len(prt)):
#             ptchs.append(Polygon(pts[par[pij]:par[pij+1]]))

# #        ax.add_collection(PatchCollection(ptchs,facecolor=cccol[nshp,:],edgecolor='k', linewidths=.1))
#         ax.add_collection(PatchCollection(ptchs,facecolor=cccol[nshp],norm=colors.PowerNorm(gamma=gamma),edgecolor='k', linewidths=.1))
#     ax.set_xlim(-91,-82)
#     ax.set_ylim(41,48)


# #    if len(colorcol)>0:
# #        c = dfl[colorcol].copy()#**gamma
# #        plt.scatter(dfl_zctacenters['longitude'],\
# #                    dfl_zctacenters['latitude'],c=c, norm=colors.PowerNorm(gamma=gamma))


# def draw_zips_3(zctashapes, dfl, colorcol='', gamma=1.0):

#     if len(colorcol)>0:
#         c = dfl[colorcol].copy()#**gamma

#     shapes = zctashapes
#     Nshp    = len(shapes)

#     cm    = plt.get_cmap('Dark2')
#     cccol = cm(1.*c/max(c))

#     #   -- plot --
#     fig     = plt.figure()
#     ax      = fig.add_subplot(111)
#     for nshp in xrange(Nshp):
#         ptchs   = []
#         pts     = np.array(shapes[nshp].points)
#         prt     = shapes[nshp].parts
#         par     = list(prt) + [pts.shape[0]]

#         for pij in xrange(len(prt)):
#             ptchs.append(Polygon(pts[par[pij]:par[pij+1]]))

# #        ax.add_collection(PatchCollection(ptchs,facecolor=cccol[nshp,:],edgecolor='k', linewidths=.1))
#         pc = PatchCollection(ptchs,facecolor=cccol[nshp],norm=colors.PowerNorm(gamma=gamma),edgecolor='k', linewidths=.1)
#         # for collection in pc:?
#         pc.set_facecolor(cccol[nshp])

#         ax.add_collection(pc)
#     ax.set_xlim(-91,-82)
#     ax.set_ylim(41,48)

#     return ax

# #    if len(colorcol)>0:
# #        c = dfl[colorcol].copy()#**gamma
# #        plt.scatter(dfl_zctacenters['longitude'],\
# #                    dfl_zctacenters['latitude'],c=c, norm=colors.PowerNorm(gamma=gamma))


# #pc = PatchCollection(patches, match_original=True)

# #norm = Normalize()
# #cmap = plt.get_cmap('Blues')
# #pc.set_facecolor(cmap(norm(values)))
# #ax.add_collection(pc)

#


