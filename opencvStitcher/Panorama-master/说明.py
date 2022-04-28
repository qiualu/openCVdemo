"""
# opencv

# detectAndCompute

# 使用函数detectAndCompute(）检测关键点并计算描述符

函数detectAndCompute(）参数说明：

void detectAndCompute(
InputArray image, //图像
InputArray mask, //掩模
CV_OUT std::vector& keypoints,//输出关键点的集合
OutputArray descriptors,//计算描述符（descriptors[i]是为keypoints[i]的计算描述符）
bool useProvidedKeypoints=false //使用提供的关键点
);

python # Find interest points and Computing features.
keypoints, features = md.detectAndCompute(grayImage, None)

match(）从查询集中查找每个描述符的最佳匹配。

参数说明：
void match(
InputArray queryDescriptors, //查询描述符集
InputArray trainDescriptors, //训练描述符集合
CV_OUT std::vector& matches, //匹配
InputArray mask=noArray() //指定输入查询和描述符的列表矩阵之间的允许匹配的掩码
) const;
————————————————
版权声明：本文为CSDN博主「caogps」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/caogps/article/details/107850859


"""












