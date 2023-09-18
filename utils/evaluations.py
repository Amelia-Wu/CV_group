import pandas as pd

top_k = 2
def evaluate(file_to_evaluation, lookup_table, ground_truth_table):
    # 读取file_to_evaluation.csv文件, 选出每行最大的两个值，记住它们的索引
    df = pd.read_csv(file_to_evaluation)
    df_values = df.iloc[:, 1:]
    df_values = df_values.apply(lambda row: row.astype(float), axis=1)
    max_values = df_values.apply(lambda row: sorted(zip(row, range(len(row))), reverse=True)[:2], axis=1)
    df["max_values"] = max_values  # [(max_value, index1), (second_max_value, index2)]

    lookup_df = pd.read_csv(lookup_table)

    # 在lookup_df中找到对应的文件名(元素与df中的位置相同)
    # Use the column indices to get the corresponding filenames
    def get_filenames(row):
        # Get the indices from max_values column
        col_indices = [item[1] for item in row['max_values']]

        # Adjust indices to match df columns (because index 0 means second column in df)
        adjusted_col_indices = [index + 1 for index in col_indices]

        # Get the filenames using the column indices
        row_index = row.name
        filenames = [lookup_df.iloc[row_index, index] for index in adjusted_col_indices]
        return filenames

    df['filenames'] = df.apply(get_filenames, axis=1)

    # 读取ground_truth_table.csv文件
    ground_truth_df = pd.read_csv(ground_truth_table)

    # ground_truth_table.csv文件中 第一列是left image的文件名，第二列是right image的文件名。将数据储存到词典里以供查询
    ground_truth_dict = {}
    for index, row in ground_truth_df.iterrows():
        ground_truth_dict[row[0]] = row[1]

    # 比较df中每个left image的filenames的元素是否包含ground_truth_dict中的文件名，如果包含，就说明预测正确，否则预测错误
    def is_prediction_correct(row):
        if ground_truth_dict[row['left']] in row['filenames']:
            return True
        else:
            return False

    df['is_prediction_correct'] = df.apply(is_prediction_correct, axis=1)

    # 计算top-k accuracy
    accuracy = df['is_prediction_correct'].sum() / len(df)
    print(accuracy)


    # 储存到csv文件
    df.to_csv("../outputFile/temp_test_evaluation.csv", index=False)




# def evaluate(file_to_evaluation, lookup_table, ground_truth_table):
#
#     # 读取file_to_evaluation
#     result = pd.read_csv(file_to_evaluation)
#     #转换c0, c1, c2, ... 为数值类型
#     result.iloc[:, 1:] = result.iloc[:, 1:].astype(float)
#
#     # 找出分数最高的两张右图片，并记住它们的索引
#     for i in range(1, top_k+1):
#         result[f'max_{i}'] = result.iloc[:, 1:].apply(lambda x: x.nlargest(i).index[-1], axis=1)
#
#     # 使用lookup_table获取文件名
#     lookup_df = pd.read_csv(lookup_table, index_col='index')
#     for i in range(1, top_k + 1):
#         result[f'filename_max_{i}'] = result[f'max_{i}'].map(lookup_df['filename'])
#
#     # 读取ground_truth_table并进行匹配
#     ground_truth_df = pd.read_csv(ground_truth_table)
#     # 假设ground_truth_table有columns: 'id' (匹配结果dataframe中的index) 和 'true_filename' (真实的文件名)
#     result = result.merge(ground_truth_df, left_index=True, right_on='id', how='left')
#
#     # 计算top-k accuracy
#     correct_count = 0
#     for idx, row in result.iterrows():
#         predicted_files = [row[f'filename_max_{i}'] for i in range(1, top_k + 1)]
#         if row['true_filename'] in predicted_files:
#             correct_count += 1
#
#     accuracy = correct_count / len(result)
#     print(accuracy)
#
#     return accuracy



if __name__ == '__main__':
    evaluate('../dataset/temp_test_submission.csv', '../dataset/extended_train.csv', '../dataset/train.csv')
