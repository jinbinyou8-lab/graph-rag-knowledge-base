from neo4j import GraphDatabase

# ============ 1. 建立连接 ============
URI = "neo4j://127.0.0.1:7690"
AUTH = ("neo4j", "12345678")
DATABASE = "neo4j"

driver = GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()
print("连接成功！")


with driver.session(database=DATABASE) as session:
    session.run("Match (n) DETACH DELETE n").consume()  #consume()就是告诉驱动直接执行就行不需要返回值

    #创建节点
    session.run(
        "CREATE (:Customer {name: $name, age: $age, vip: $vip})",
        name = "张三", age=28, vip=True
    ).consume()