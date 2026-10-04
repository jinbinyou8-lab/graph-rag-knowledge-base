from langgraph.graph import StateGraph
from langgraph.constants import START, END
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.nodes.pdf_to_md_mode import Pdf_To_Md_Node
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.md_img_node import MarkDownImageNode
from knowledge.processor.import_process.state import create_default_state
from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.nodes.document_spliter_node import DocumentSpliterNode
from knowledge.processor.import_process.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.bge_embedding_chunks_node import BgeEmbeddingChunksNode
from knowledge.processor.import_process.nodes.import_milvus_node import ImportMilvusNode
from knowledge.processor.import_process.nodes.kg_graph_node import KnowLedgeGraphNode
def import_router(state: ImportGraphState):

    if state.get("is_md_read_enabled"):
        return "md_img_node"
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"
    return END


def create_import_graph() -> StateGraph:
    """
    定义整个导入业务的graph状态拓扑图(langgraph构建流水线) 整个流水线各个节点要读取或者写入的节点
    Returns:

    """

    #1. 定义状态图
    graph_pipeline = StateGraph(ImportGraphState)    # type:ignore

    #2. 定义节点(入口,结束节点,自己需要添加的)
    #2.1 定义入口节点

    graph_pipeline.set_entry_point("entry_node")

    #2.2 定义剩下的节点
    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": Pdf_To_Md_Node(),
        "md_img_node": MarkDownImageNode(),
        "document_split": DocumentSpliterNode(),
        "item_name_node":ItemNameRecognitionNode(),
        "bge_embedding_chunks": BgeEmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode(),
        "kg_graph_node": KnowLedgeGraphNode()
    }
    for key, value in nodes.items():
        graph_pipeline.add_node(key, value)

    #3. 定义边(顺序边,条件边)
    # 条件边
    #source: 路由开始节点
    #path: 路由函数
    #path_map: 路由函数的映射

    graph_pipeline.add_conditional_edges("entry_node",
                                         import_router,
                                         {
                                             "md_img_node": "md_img_node",
                                             "pdf_to_md_node":"pdf_to_md_node",
                                             END: END
                                         }
                                         )

    graph_pipeline.add_edge("pdf_to_md_node", "md_img_node")
    graph_pipeline.add_edge("md_img_node", "document_split")
    graph_pipeline.add_edge("document_split", "item_name_node")
    graph_pipeline.add_edge( "item_name_node", "bge_embedding_chunks")
    graph_pipeline.add_edge("bge_embedding_chunks", "import_milvus_node")
    graph_pipeline.add_edge( "import_milvus_node", "kg_graph_node")
    graph_pipeline.add_edge("kg_graph_node", END)



    #4. 编译(编排)
    return graph_pipeline.compile()


kb_import_graph_app = create_import_graph()

# 测试调用
def run_import_graph(import_file_path: str, file_dir: str):

    #1. 构造state
    state = {
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }
    init_state = create_default_state(**state)      #发生了解包,这里需要再理解一下,目前其实不用这个也行,可以直接传递state

    #2. 调用stream(用流式获取每一个节点的处理情况: event时间[节点名字 节点处理后的状态])
    for event in kb_import_graph_app.stream(init_state):
        print(event)


if __name__ == "__main__":
    setup_logging()

    import_file_path = r"F:\agent_project_knowledgestore\shopkeeper_brain\knowledge\processor\import_process\万用表的使用\hybrid_auto\万用表的使用.md"
    file_dir = r"F:\agent_project_knowledgestore\shopkeeper_brain\knowledge\processor\import_process"
    run_import_graph(import_file_path=import_file_path,file_dir=file_dir)

    #4. 打印图结果 (ASCII 可视化)
    print("=="*50)
    print("图结构:")
    kb_import_graph_app.get_graph().print_ascii()         #展示图结构