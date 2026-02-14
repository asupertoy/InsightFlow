from typing import List, Dict, Any, Optional, Type
from datetime import datetime
import json
import uuid
import re
from pathlib import Path
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool

# 获取项目根目录 (假设此文件在 insightflow_core/tools/note_tool.py)
# parents[0] -> tools
# parents[1] -> insightflow_core
# parents[2] -> InsightFlow (Project Root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class NoteToolInput(BaseModel):
    action: str = Field(
        ...,
        description="操作类型: create(创建), read(读取), update(更新), delete(删除), list(列表), search(搜索), summary(摘要)"
    )
    title: Optional[str] = Field(None, description="笔记标题（create/update时必需）")
    content: Optional[str] = Field(None, description="笔记内容（create/update时必需）")
    note_type: str = Field(
        "general",
        description="笔记类型: task_state(任务状态), conclusion(结论), blocker(阻塞项), action(行动计划), reference(参考), general(通用)"
    )
    tags: Optional[List[str]] = Field(None, description="标签列表（可选）")
    note_id: Optional[str] = Field(None, description="笔记ID（read/update/delete时必需）")
    query: Optional[str] = Field(None, description="搜索关键词（search时必需）")
    limit: int = Field(10, description="返回结果数量限制（默认10）")

class NoteTool(BaseTool):
    name: str = "note_tool"
    description: str = (
        "用于管理工作记忆和长期记忆的笔记工具。支持创建、读取、更新、删除、搜索笔记。"
        "当你需要记录任务状态、保存重要结论、列出待办事项或整理思路时，请使用此工具。"
    )
    args_schema: Type[BaseModel] = NoteToolInput
    
    # Instance attributes
    _base_dir: Path = PrivateAttr(default=PROJECT_ROOT / "data" / "notes")
    _notes_dir: Path = PrivateAttr(default=PROJECT_ROOT / "data" / "notes" / "content")
    _index_file: Path = PrivateAttr(default=PROJECT_ROOT / "data" / "notes" / "notes_index.json")
    _max_notes: int = PrivateAttr(default=1000)
    _notes_index: Dict[str, Any] = PrivateAttr(default_factory=dict)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize storage
        self._init_storage()

    def _init_storage(self):
        """初始化存储目录和索引文件"""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._notes_dir.mkdir(parents=True, exist_ok=True)
        
        if not self._index_file.exists():
            initial_index = {
                "metadata": {
                    "version": "1.0",
                    "created_at": datetime.now().isoformat(),
                    "total_notes": 0
                },
                "notes": []
            }
            self._save_index(initial_index)
        
    def _load_index(self) -> Dict[str, Any]:
        """加载索引"""
        try:
            with open(self._index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            # Fallback if corrupted
            return {"metadata": {"total_notes": 0}, "notes": []}

    def _save_index(self, index_data: Dict[str, Any]):
        """保存索引"""
        with open(self._index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

    def _generate_note_id(self) -> str:
        """生成短ID"""
        return str(uuid.uuid4())[:8]

    def _get_note_path(self, note_id: str) -> Path:
        """获取笔记文件路径"""
        return self._notes_dir / f"{note_id}.md"

    def _note_to_markdown(self, note: Dict[str, Any]) -> str:
        """将笔记对象转换为Markdown格式（带Frontmatter）"""
        frontmatter = {
            "id": note["id"],
            "title": note["title"],
            "type": note["type"],
            "tags": note.get("tags", []),
            "created_at": note["created_at"],
            "updated_at": note["updated_at"],
            "metadata": note.get("metadata", {})
        }
        
        fm_str = json.dumps(frontmatter, indent=2, ensure_ascii=False)
        return f"---\n{fm_str}\n---\n\n{note['content']}"

    def _markdown_to_note(self, markdown_text: str) -> Dict[str, Any]:
        """解析Markdown笔记文件"""
        pattern = r"^---\n(.*?)\n---\n\n(.*)$"
        match = re.search(pattern, markdown_text, re.DOTALL)
        
        if match:
            fm_str = match.group(1)
            content = match.group(2)
            try:
                note = json.loads(fm_str)
                note["content"] = content
                return note
            except json.JSONDecodeError:
                pass
        
        # Fallback parsing
        return {
            "id": "unknown",
            "title": "Unknown",
            "content": markdown_text,
            "type": "general",
            "created_at": "",
            "updated_at": ""
        }

    def get_note(self, note_id: str) -> Optional[Dict[str, Any]]:
        """
        获取笔记的结构化数据（字典格式）。
        供其他节点（如Planner）编程调用，而非LLM直接调用。
        """
        if not note_id:
            return None
        
        note_path = self._get_note_path(note_id)
        if not note_path.exists():
            return None
            
        try:
            with open(note_path, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
            return self._markdown_to_note(markdown_text)
        except Exception as e:
            print(f"Error reading note file: {e}")
            return None

    def _run(self, action: str, title: Optional[str] = None, content: Optional[str] = None, 
             note_type: str = "general", tags: Optional[List[str]] = None, 
             note_id: Optional[str] = None, query: Optional[str] = None, limit: int = 10, **kwargs) -> str:
        
        # Reload index on every run to ensure freshness
        self._notes_index = self._load_index()

        if action == "create":
            return self._create_note(title, content, note_type, tags, note_id)
        elif action == "read":
            return self._read_note(note_id)
        elif action == "update":
            return self._update_note(note_id, title, content, note_type, tags)
        elif action == "delete":
            return self._delete_note(note_id)
        elif action == "list":
            return self._list_notes(note_type, limit)
        elif action == "search":
            return self._search_notes(query, limit)
        elif action == "summary":
            return self._get_summary()
        else:
            return f"❌ 不支持的操作: {action}"

    def _create_note(self, title, content, note_type, tags, note_id=None) -> str:
        if not title or not content:
            return "❌ 创建笔记需要提供 title 和 content"
        
        if len(self._notes_index["notes"]) >= self._max_notes:
            return f"❌ 笔记数量已达上限 ({self._max_notes})"
        
        if not note_id:
            note_id = self._generate_note_id()
        
        note = {
            "id": note_id,
            "title": title,
            "content": content,
            "type": note_type,
            "tags": tags if isinstance(tags, list) else [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {
                "word_count": len(content),
                "status": "active"
            }
        }
        
        # Save file
        note_path = self._get_note_path(note_id)
        markdown_content = self._note_to_markdown(note)
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # Update index
        self._notes_index["notes"].append({
            "id": note_id,
            "title": title,
            "type": note_type,
            "tags": tags if isinstance(tags, list) else [],
            "created_at": note["created_at"]
        })
        self._notes_index["metadata"]["total_notes"] = len(self._notes_index["notes"])
        self._save_index(self._notes_index)
        
        return f"✅ 笔记创建成功\nID: {note_id}\n标题: {title}\n类型: {note_type}"
    
    def _read_note(self, note_id) -> str:
        if not note_id:
            return "❌ 读取笔记需要提供 note_id"
        
        note_path = self._get_note_path(note_id)
        if not note_path.exists():
            return f"❌ 笔记不存在: {note_id}"
        
        with open(note_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        
        note = self._markdown_to_note(markdown_text)
        return self._format_note(note)

    def _update_note(self, note_id, title, content, note_type, tags) -> str:
        if not note_id:
            return "❌ 更新笔记需要提供 note_id"
        
        note_path = self._get_note_path(note_id)
        if not note_path.exists():
            return f"❌ 笔记不存在: {note_id}"
        
        with open(note_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        note = self._markdown_to_note(markdown_text)
        
        if title: note["title"] = title
        if content:
            note["content"] = content
            note["metadata"]["word_count"] = len(content)
        if note_type: note["type"] = note_type
        if tags is not None: note["tags"] = tags
        
        note["updated_at"] = datetime.now().isoformat()
        
        markdown_content = self._note_to_markdown(note)
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        for idx_note in self._notes_index["notes"]:
            if idx_note["id"] == note_id:
                if title: idx_note["title"] = title
                if note_type: idx_note["type"] = note_type
                if tags is not None: idx_note["tags"] = tags
                break
        self._save_index(self._notes_index)
        
        return f"✅ 笔记更新成功: {note_id}"

    def _delete_note(self, note_id) -> str:
        if not note_id:
            return "❌ 删除笔记需要提供 note_id"
        
        note_path = self._get_note_path(note_id)
        if note_path.exists():
            note_path.unlink()
        
        self._notes_index["notes"] = [
            n for n in self._notes_index["notes"] if n["id"] != note_id
        ]
        self._notes_index["metadata"]["total_notes"] = len(self._notes_index["notes"])
        self._save_index(self._notes_index)
        
        return f"✅ 笔记已删除: {note_id}"

    def _list_notes(self, note_type, limit) -> str:
        filtered_notes = self._notes_index["notes"]
        if note_type:
            filtered_notes = [n for n in filtered_notes if n["type"] == note_type]
        
        filtered_notes = filtered_notes[:limit]
        
        if not filtered_notes:
            return "📝 暂无笔记"
        
        result = f"📝 笔记列表（共 {len(filtered_notes)} 条）\n\n"
        for note in filtered_notes:
            result += f"• [{note['type']}] {note['title']}\n"
            result += f"  ID: {note['id']}\n"
            if note.get('tags'):
                result += f"  标签: {', '.join(note['tags'])}\n"
            result += f"  创建时间: {note['created_at']}\n\n"
        
        return result

    def _search_notes(self, query, limit) -> str:
        if not query:
            return "❌ 搜索需要提供 query"
        
        query = query.lower()
        matched_notes = []
        for idx_note in self._notes_index["notes"]:
            note_path = self._get_note_path(idx_note["id"])
            if note_path.exists():
                try:
                    with open(note_path, 'r', encoding='utf-8') as f:
                        markdown_text = f.read()
                    note = self._markdown_to_note(markdown_text)
                    
                    if (query in note["title"].lower() or
                        query in note["content"].lower() or
                        any(query in tag.lower() for tag in note.get("tags", []))):
                        matched_notes.append(note)
                except Exception:
                    continue
        
        matched_notes = matched_notes[:limit]
        
        if not matched_notes:
            return f"📝 未找到匹配 '{query}' 的笔记"
        
        result = f"🔍 搜索结果（共 {len(matched_notes)} 条）\n\n"
        for note in matched_notes:
            result += self._format_note(note, compact=True) + "\n"
        
        return result

    def _get_summary(self) -> str:
        total = len(self._notes_index["notes"])
        type_counts = {}
        for note in self._notes_index["notes"]:
            note_type = note["type"]
            type_counts[note_type] = type_counts.get(note_type, 0) + 1
        
        result = f"📊 笔记摘要\n\n"
        result += f"总笔记数: {total}\n\n"
        result += "按类型统计:\n"
        for note_type, count in sorted(type_counts.items()):
            result += f"  • {note_type}: {count}\n"
        
        return result

    def _format_note(self, note: Dict[str, Any], compact: bool = False) -> str:
        if compact:
            content_preview = note['content'][:100].replace('\n', ' ')
            return (
                f"[{note['type']}] {note['title']}\n"
                f"ID: {note['id']} | 内容: {content_preview}..."
            )
        else:
            result = f"📝 笔记详情\n\n"
            result += f"ID: {note['id']}\n"
            result += f"标题: {note['title']}\n"
            result += f"类型: {note['type']}\n"
            if note.get('tags'):
                result += f"标签: {', '.join(note['tags'])}\n"
            result += f"创建时间: {note['created_at']}\n"
            result += f"更新时间: {note['updated_at']}\n"
            result += f"\n内容:\n{note['content']}\n"
            return result
