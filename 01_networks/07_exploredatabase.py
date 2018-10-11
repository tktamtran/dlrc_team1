
import sys
sys.path.insert(0, '/home/dlrc1/Desktop/00_introstuff')
from utils import *


directory_data = 'data_robot/'
database = pd.read_pickle(directory_data + 'datawcs_robot_batches0020.pkl')

# concatenate the batches of data
if False:
    directory_data = 'data_robot/'
    files_data = [f for f in listdir(directory_data) if 'datawcs_robot' in f]

    data_all = pd.read_pickle(directory_data + files_data[0])

    for e,file_data in enumerate(files_data[1:]):
        data_all = pd.concat([data_all, pd.read_pickle(directory_data + file_data)])

    print(data_all.shape)
    input('..')

    data_all.to_pickle(directory_data + 'datawcs_robot_batches0020.pkl')



# explore range and distribution of joints of collected data
if False:
    cols_joint = ['j' + str(j) for j in np.arange(1, 8)]
    ranges = np.array([(min(database[j]), max(database[j])) for j in cols_joint])
    print(ranges)

    # plt.ion()
    fig = plt.figure()
    subplot_row, subplot_column = 4, 2
    for i, j in enumerate(cols_joint):
        ax = fig.add_subplot(subplot_row, subplot_column, i + 1)
        ax.hist(database[j], bins=100)
        ax.set_title(j)

    plt.show()



#rename dataframe columns
if False:
    cols_dict = {}
    for i,j in enumerate(cols_joint):
        cols_dict[j] = 'j'+str(i+1)

    database = database.rename(columns = cols_dict)
    print(database.columns)
    input('..')
    database.to_pickle(directory_data + 'datawcs_robot_batches0020.pkl')




#hist, bins = np.histogram(database['j1'], bins=100)
#print(bins)

query_joint = [0.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0]
query_joint = [-2.696016, -0.141817,-2.163907, -2.104302, -0.035638,  0.900234, -2.717362]
query_joint = [-0.074236,  0.194738,  0.988106, -2.043999,  0.051827,  0.591738, -2.192394]
query_joint = [0.015598,  0.033946,  0.178241, -1.552231, -0.013004,  1.310330,  0.636937]
db_matches = (np.abs(database['j1'].values - query_joint[0]) < 0.1) * \
             (np.abs(database['j2'].values - query_joint[1]) < 0.1) * \
             (np.abs(database['j3'].values - query_joint[2]) < 0.1) * \
             (np.abs(database['j4'].values - query_joint[3]) < 0.1) * \
             (np.abs(database['j5'].values - query_joint[4]) < 0.1) * \
             (np.abs(database['j6'].values - query_joint[5]) < 0.1) * \
             (np.abs(database['j7'].values - query_joint[6]) < 0.1)
idx_matches = np.argwhere(db_matches).squeeze(axis=1)
print('n_wcs_matches', idx_matches.shape)
print('n_joint_matches') #TODO


