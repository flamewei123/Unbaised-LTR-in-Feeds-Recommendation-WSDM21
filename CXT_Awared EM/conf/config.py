# this file is used for define constant parameter

from itertools import product
import pandas as pd
import os


feature_135 = [
'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15', 'feature_16', 'feature_17', 'feature_18', 'feature_19', 'feature_20', 'feature_21', 'feature_22', 'feature_23', 'feature_24', 'feature_25', 'feature_26', 'feature_27', 'feature_28', 'feature_29', 'feature_30', 'feature_31', 'feature_32', 'feature_33', 'feature_34', 'feature_35', 'feature_36', 'feature_37', 'feature_38', 'feature_39', 'feature_40', 'feature_41', 'feature_42', 'feature_43', 'feature_44', 'feature_45', 'feature_46', 'feature_47', 'feature_48', 'feature_49', 'feature_50', 'feature_51', 'feature_52', 'feature_53', 'feature_54', 'feature_55', 'feature_56', 'feature_57', 'feature_58', 'feature_59', 'feature_60', 'feature_61', 'feature_62', 'feature_63', 'feature_64', 'feature_65', 'feature_66', 'feature_67', 'feature_68', 'feature_69', 'feature_70', 'feature_71', 'feature_72', 'feature_73', 'feature_74', 'feature_75', 'feature_76', 'feature_77', 'feature_78', 'feature_79', 'feature_80', 'feature_81', 'feature_82', 'feature_83', 'feature_84', 'feature_85', 'feature_86', 'feature_87', 'feature_88', 'feature_89', 'feature_90', 'feature_91', 'feature_92', 'feature_93', 'feature_94', 'feature_95', 'feature_96', 'feature_97', 'feature_98', 'feature_99', 'feature_100', 'feature_101', 'feature_102', 'feature_103', 'feature_104', 'feature_105', 'feature_106', 'feature_107', 'feature_108', 'feature_109', 'feature_110', 'feature_111', 'feature_112', 'feature_113', 'feature_114', 'feature_115', 'feature_116', 'feature_117', 'feature_118', 'feature_119', 'feature_120', 'feature_121', 'feature_122', 'feature_123', 'feature_124', 'feature_125', 'feature_126', 'feature_127', 'feature_128', 'feature_129', 'feature_130', 'feature_131', 'feature_132', 'feature_133', 'feature_134', 'feature_135'
]
feature_177 = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14', 'feature15', 'feature16', 'feature17', 'feature18', 'feature19', 'feature20', 'feature21', 'feature22', 'feature23', 'feature24', 'feature25', 'feature26', 'feature27', 'feature28', 'feature29', 'feature30', 'feature31', 'feature32', 'feature33', 'feature34', 'feature35', 'feature36', 'feature37', 'feature38', 'feature39', 'feature40', 'feature41', 'feature42', 'feature43', 'feature44', 'feature45', 'feature46', 'feature47', 'feature48', 'feature49', 'feature50', 'feature51', 'feature52', 'feature53', 'feature54', 'feature55', 'feature56', 'feature57', 'feature58', 'feature59', 'feature60', 'feature61', 'feature62', 'feature63', 'feature64', 'feature65', 'feature66', 'feature67', 'feature68', 'feature69', 'feature70', 'feature71', 'feature72', 'feature73', 'feature74', 'feature75', 'feature76', 'feature77', 'feature78', 'feature79', 'feature80', 'feature81', 'feature82', 'feature83', 'feature84', 'feature85', 'feature86', 'feature87', 'feature88', 'feature89', 'feature90', 'feature91', 'feature92', 'feature93', 'feature94', 'feature95', 'feature96', 'feature97', 'feature98', 'feature99', 'feature100', 'feature101', 'feature102', 'feature103', 'feature104', 'feature105', 'feature106', 'feature107', 'feature108', 'feature109', 'feature110', 'feature111', 'feature112', 'feature113', 'feature114', 'feature115', 'feature116', 'feature117', 'feature118', 'feature119', 'feature120', 'feature121', 'feature122', 'feature123', 'feature124', 'feature125', 'feature126', 'feature127', 'feature128', 'feature129', 'feature130', 'feature131', 'feature132', 'feature133', 'feature134', 'feature135', 'feature136', 'feature137', 'feature138', 'feature139', 'feature140', 'feature141', 'feature142', 'feature143', 'feature144', 'feature145', 'feature146', 'feature147', 'feature148', 'feature149', 'feature150', 'feature151', 'feature152', 'feature153', 'feature154', 'feature155', 'feature156', 'feature157', 'feature158', 'feature159', 'feature160', 'feature161', 'feature162', 'feature163', 'feature164', 'feature165', 'feature166', 'feature167', 'feature168', 'feature169', 'feature170', 'feature171', 'feature172', 'feature173', 'feature174', 'feature175', 'feature176', 'feature177']


trainfeature = feature_135 + feature_177 

schema = ['arrival_id', 'cid3_brand', 'click', 'pos', 'with_high_similarity']

##### the data we used has feature names, we donot change this for convenience, you should rewrite this part for your dataset 

schema.extend(trainfeature)

# tree parameters
tree_num = 100

# version num
row_num = iter_df.shape[0]
version_num = row_num

# data
file_path = os.path.dirname(os.path.abspath(__file__)) + "/../"

# other parameters
# !!
is_use_pairwise_training = False



if __name__=='__main__':

    print len(trainfeature)




