import pandas as pd
import matplotlib.pyplot as plt
import os

def parse_v2_label(x):
    if len(x) == 0:
        return '空'
    else:
        return x[0]['name']


def convert_float_str(x):
    try:
        return str(float(x))
    except:
        return str(x)
def draw_discrete_bar(sta_dict, title, layout):
    row, col, index = layout
    plt.subplot(row, col, index)
    # discrete_label = discrete_cls_list[title]
    # another_label = []
    # for i in sta_dict:
    #     if i not in discrete_label:
    #         another_label.append(i)
    # discrete_label += another_label
    discrete_label = list(sta_dict.keys())
    plt.bar(range(len(discrete_label)), [sta_dict.get(xtick, 0) for xtick in discrete_label], align='center')
    plt.xticks(range(len(discrete_label)), discrete_label, rotation=40)
    plt.title(title)

def draw_discrete(discrete, save_file, dst_columns, row, col, fillna_v='空', v2=False):
    fig = plt.figure(figsize=(30,60))
    if not v2:
        discrete_df = discrete.fillna(fillna_v).astype(str)
    else:
        discrete_df = discrete

    for i in range(len(dst_columns)):
        if not v2:
            data_dict = discrete_df[dst_columns[i]].value_counts(dropna=False).to_dict()
        else:
            data_dict = discrete_df[dst_columns[i]].apply(parse_v2_label).value_counts(dropna=False).to_dict()
        draw_discrete_bar(data_dict, dst_columns[i], (row, col, i+1))
    plt.savefig(os.path.join("./", save_file))

# path = '/opt/tiger/mlx_notebook/cc/classification/video/data/data_v1/vu_jy_tag_01.parquet'
# df = pd.read_parquet(path, engine='pyarrow')
# draw_discrete(df, "label_distribute.jpg", ['category', 'style', 'expression', 'person'], 4, 1, fillna_v='空')

path = '/opt/tiger/mlx_notebook/cc/classification/video/data/data_v2/info.parquet'
df = pd.read_parquet(path, engine='pyarrow')
draw_discrete(df, "label_distribute_v2.jpg", ['label_person', 'label_scene', 'label_style', 'label_expression', 'label_material'], 5, 1, fillna_v='空', v2=True)


# for index, row in df.iterrows():
#     if row['style'] == "未知":
#         print(row)
#         break
#
#
#     print(index)
#     print("====")
#     print(row)
#     break