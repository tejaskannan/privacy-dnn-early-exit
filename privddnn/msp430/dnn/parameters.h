#include <stdint.h>
#include "matrix.h"
#ifndef PARAMETERS_H_
#define PARAMETERS_H_
#define EXIT_RATE 19660
#define THRESHOLD 1014
#define IS_MAX_PROB
#define IS_DNN
#define PRECISION 10
#define NUM_LABELS 10
#define NUM_INPUT_FEATURES 16

#pragma PERSISTENT(W0_DATA)
static int16_t W0_DATA[384] = { 121,296,172,-12,-13,-43,-461,850,-310,1175,673,179,182,262,-332,186,-725,344,450,352,-412,686,41,380,915,-196,-168,78,-601,-293,263,10,497,172,-618,858,-187,-499,343,-512,146,-235,-145,-71,141,388,26,1461,-40,-774,-146,-291,153,-46,-256,630,-348,-294,-498,-219,-304,-851,412,-830,509,582,349,-145,-49,-14,-846,-461,-411,-974,-47,-374,170,-113,-491,337,-254,632,-671,-123,850,188,137,168,-263,-26,-535,262,-87,573,117,979,386,65,-505,585,-106,295,859,224,-166,-285,-324,-599,456,-81,162,521,-89,-725,-80,448,647,191,-202,250,141,618,716,-530,-9,-699,-187,-133,403,-541,241,-311,-620,726,-552,564,392,-802,-343,-509,-448,19,-439,-249,-440,121,27,36,58,-500,329,-571,621,-361,-239,-250,196,27,-550,-206,-637,292,-57,-53,-627,-244,-232,-593,607,-800,593,158,414,212,238,-339,-297,715,-298,604,-506,148,-168,-276,372,-426,412,-177,-174,828,-158,372,-471,338,63,712,244,102,272,-106,163,-224,-193,-806,-43,25,-369,386,43,-83,235,-167,163,-1055,317,-691,-31,374,261,793,786,877,-831,463,-498,-315,248,-874,218,120,311,521,395,834,186,463,141,415,-160,-255,-500,783,201,366,354,190,94,26,-723,-63,-197,578,-544,51,207,-469,-169,-298,252,-787,85,-173,466,41,738,334,-332,518,161,172,-189,-806,-446,71,128,-84,437,-22,-315,167,-398,-47,387,-289,38,98,-731,85,-548,-359,-32,-661,161,-359,516,-228,-254,16,-1,827,-880,915,853,489,847,671,116,-315,-822,-384,-127,-3,-6,55,569,-383,94,-846,-432,-361,-277,310,-712,280,486,-150,385,-160,-199,-828,44,-201,-438,95,855,-1038,809,140,214,521,308,244,-243,568,408,991,892,369,575,-94,185,-486,124,392,-360,160,-28,-98,-132,-324,-172,472,96,-46,277,-796,131,-1053,413,-117,-256,21,-559,-24,172,960,643,670,503,150,274,199,360,-343 };
static struct matrix W0 = { W0_DATA, 24, 16 };

#pragma PERSISTENT(B0_DATA)
static int16_t B0_DATA[48] = { 131,0,491,0,178,0,74,0,211,0,625,0,-176,0,-43,0,573,0,145,0,7,0,539,0,353,0,-34,0,238,0,-289,0,527,0,306,0,42,0,195,0,572,0,212,0,212,0,41,0 };
static struct matrix B0 = { B0_DATA, 24, 2 };

#pragma PERSISTENT(W1_DATA)
static int16_t W1_DATA[240] = { -133,-832,991,77,-6,-476,216,-431,-394,419,492,346,86,33,421,-656,917,-749,-745,383,-1185,-57,104,-516,-1270,629,-792,654,36,-1819,-536,-169,1009,-287,45,-987,-222,-1431,717,-222,1006,-369,-1050,-158,264,-504,205,-67,-1277,-498,-851,810,-493,-1290,811,-118,-1767,-977,-553,-596,-85,-126,-682,309,-57,-570,-469,-224,666,-1065,778,-295,-1412,200,-888,-775,-975,-669,-413,709,-1211,755,-888,-275,598,-1196,-910,-761,-829,113,-38,-202,-41,420,702,-956,-75,375,-1244,-1260,24,-789,-1158,-1694,-971,-166,632,210,-730,-1286,-430,205,706,-711,-721,420,-45,-229,648,478,400,172,822,-862,39,1112,-832,212,331,-1146,-1010,185,147,-1031,-1632,575,-1018,-249,1030,515,-1144,245,39,-615,-625,-1171,-951,-668,988,-1028,-367,-1491,590,319,273,428,127,-1133,134,-570,-592,24,50,687,-538,-1221,13,-741,106,613,-1364,-1406,-250,149,566,-1071,-1332,-346,-99,927,645,-415,-147,389,-429,-327,-849,-914,367,-292,-769,514,1375,138,645,629,502,601,1064,392,966,-576,-1030,52,321,971,-48,-1574,-946,490,-880,-1007,-697,-1246,-781,-247,906,-427,-1037,-432,-1621,845,416,14,365,-386,-512,-450,-765,-179,-610,-378,-494,184,996,386,-744,825,-110,893 };
static struct matrix W1 = { W1_DATA, 10, 24 };

#pragma PERSISTENT(B1_DATA)
static int16_t B1_DATA[20] = { -54,0,148,0,602,0,-234,0,275,0,40,0,-154,0,644,0,481,0,171,0 };
static struct matrix B1 = { B1_DATA, 10, 2 };

#endif
