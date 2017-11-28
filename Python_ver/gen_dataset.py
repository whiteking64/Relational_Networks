"""
データセットの作成・保存
"""
import os
import random
import pickle
import numpy as np
import cv2

train_size = 9800
test_size = 200
img_size = 75
size = 5
question_size = 11 ##6 for one-hot vector of color, 2 for question type, 3 for question subtype
"""Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""

#問題はそれぞれ10問
nb_questions = 10
dirs = './data'

colors = [
    (0, 0, 255), #r
    (0, 255, 0), #g
    (255, 0, 0), #b
    (0, 156, 255), #o
    (128, 128, 128), #k
    (0, 255, 255) #y
]

try:
    os.makedirs(dirs)
except:
    print('directory {} already exists'.format(dirs))

def center_generate(objects):
    """
    sizeは図形の大きさを意味している？
    objectの長さが正である限り，whileを抜けない
    図形同士が重なるとまずいため
    もし重ならなかったら，pasをTrueにして，whileを抜ける．生成されたcenterを返す．
    """
    while True:
        pas = True
        center = np.random.randint(0+size, img_size - size, 2) #プロット用の中心座標を生成？
        if len(objects) > 0:
            for name, c, shape in objects:
                if ((center - c) ** 2).sum() < ((size * 2) ** 2):
                    pas = False
        if pas:
            return center

def build_dataset():
    """
    オブジェクトの各要素はタプルになっている
    (色集合のインデックス， 色， 形（'r' or 'c'）)
    """
    objects = [] #6つのランダムな形の図形でそれぞれ違う色の図形を入れるリスト
    img = np.ones((img_size, img_size, 3)) * 255 #CNNへの入力用に画像を生成
    for color_id, color in enumerate(colors):
        center = center_generate(objects)
        if random.random() < 0.5:
            #長方形の描画
            #図形の描画のスタート位置，終了位置
            start = (center[0]-size, center[1]-size)
            end = (center[0]+size, center[1]+size)
            cv2.rectangle(img, start, end, color, -1)
            objects.append((color_id, center, 'r'))
        else:
            #円の描画
            center_ = (center[0], center[1])
            cv2.circle(img, center_, size, color, -1)
            objects.append((color_id, center, 'c'))


    rel_questions = []
    norel_questions = []
    rel_answers = []
    norel_answers = []
    #Non-relational questions
    for _ in range(nb_questions):
        question = np.zeros((question_size)) #初期化

        color = random.randint(0, 5) #この場合，5も含むのでnumpyのrandomと注意
        question[color] = 1 #colorのone-hot
        question[6] = 1 #non_relation
        subtype = random.randint(0, 2) #subtypeインデックスの生成
        question[subtype+8] = 1 #subtypeのonehot
        #quiestion : [r, g, b, o, k, y, non_relation, relation, subtype0, ...]
        norel_questions.append(question)

        #answer(正解データ)のonehotベクトルの生成
        #Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]
        if subtype == 0:
            #query shape->rectangle/circle
            if objects[color][2] == 'r': #図形が長方形だった場合
                answer = 2
            else:
                answer = 3

        elif subtype == 1:
            #query horizontal position->yes/no
            if objects[color][1][0] < img_size / 2: #指定のカラーの図形が全体の写真の左にある場合
                answer = 0
            else:
                answer = 1

        elif subtype == 2:
            #query vertical position->yes/no
            if objects[color][1][1] < img_size / 2: #全体の写真の上にある場合
                answer = 0
            else:
                answer = 1
        norel_answers.append(answer)

    #Relational questions
    for _ in range(nb_questions):
        question = np.zeros((question_size)) #長さ11のベクトル
        color = random.randint(0, 5)
        question[color] = 1
        question[7] = 1#relation
        subtype = random.randint(0, 2)
        question[subtype+8] = 1
        rel_questions.append(question)

        #subtype != 2で場合分けすると少し綺麗になるかも
        if subtype == 0:
            #closest-to->rectangle/circle
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            dist_list[dist_list.index(0)] = 999
            closest = np.argmin(dist_list)
            if objects[closest][2] == 'r':
                answer = 2
            else:
                answer = 3

        elif subtype == 1:
            #furthest-from->rectangle/circle
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            furthest = np.argmax(dist_list)
            if objects[furthest][2] == 'r':
                answer = 2
            else:
                answer = 3

        elif subtype == 2:
            #count->1~6
            my_obj = objects[color][2]
            count = -1
            for obj in objects:
                if obj[2] == my_obj:
                    count += 1
            answer = count+4

        rel_answers.append(answer)

    #rel/non_relをタプルでリストに追加
    relations = (rel_questions, rel_answers)
    norelations = (norel_questions, norel_answers)

    img = img/255.
    dataset = (img, relations, norelations)
    return dataset

#train data, test_dataの作成を実行
print('building test datasets...')
test_datasets = [build_dataset() for _ in range(test_size)]
print('building train datasets...')
train_datasets = [build_dataset() for _ in range(train_size)]

print('saving datasets...')
filename = os.path.join(dirs, 'sort-of-clevr.pickle')

with  open(filename, 'wb') as f:
    pickle.dump((train_datasets, test_datasets), f) #データセットをバイト列にして書き出す
print('datasets saved at {}'.format(filename))
