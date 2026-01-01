from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

# 读取图像
file_path = r"C:\Users\lenovo\Desktop\report\1.jpg"
img = Image.open(file_path)

# 增强颜色（整体饱和度）
enhancer = ImageEnhance.Color(img)
img_enhanced = enhancer.enhance(3.0)  # 颜色更加强烈

# 显示结果
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Color Enhanced (x3.0)")
plt.imshow(img_enhanced)
plt.axis("off")
plt.show()

# 保存结果
save_path = r"C:\Users\lenovo\Desktop\report\cj\fir\tree_enhanced.jpg"
img_enhanced.save(save_path)
