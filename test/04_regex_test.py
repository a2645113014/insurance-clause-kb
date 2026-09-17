import re

s = "这是第一张图![风景](a.jpg)，这是第二张图![人物](b.jpg)，这是第三张图![动物](c.jpg)"

# 正则表达式使用方式1：
# re.match("正则表达式", s)
# 正则表达式使用方式2：
# pattern = re.compile("正则表达式")
# pattern.match(s)

# r"!\[.*\]\(.*\)"
pattern = re.compile(r"!\[.*?\]\(.*?\)")
# print(pattern.findall(s))
results = pattern.finditer(s)
for match in results:
    print(match)

# 测试re.escape()
print(re.escape(s))