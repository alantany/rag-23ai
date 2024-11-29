# 基于Oracle 23c的文档问答系统

这是一个使用Oracle 23c向量数据库功能的文档问答系统。系统可以将上传的文档向量化并存储在Oracle数据库中,然后通过语义搜索找到与问题最相关的文档片段。

## 功能特点

- 支持上传PDF、Word和文本文档
- 使用Oracle 23c的向量搜索功能进行语义搜索
- 基于Streamlit构建的简洁用户界面
- 支持文档的分块处理和向量化

## 安装要求

1. Python 3.8+
2. Oracle 23c数据库
3. python-oracledb驱动
4. 其他Python依赖(见requirements.txt)

## 安装步骤

1. 克隆代码库:
```bash
git clone [repository-url]
cd [repository-name]
```

2. 安装Python依赖:
```bash
pip install -r requirements.txt
```

3. 配置Oracle数据库连接:
   - 复制.env.example为.env
   - 在.env中填入Oracle数据库的连接信息

4. 运行应用:
```bash
streamlit run main.py
```

## 使用说明

1. 启动应用后,在浏览器中打开显示的URL
2. 使用文件上传功能上传文档
3. 在输入框中输入问题
4. 系统会显示最相关的文档片段

## 注意事项

- 确保Oracle 23c数据库已正确安装和配置
- 确保数据库用户有足够的权限创建表和索引
- 建议在生产环境中使用安全的数据库连接方式
