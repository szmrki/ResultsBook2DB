import json
import urllib.request
import urllib.error
from packaging.version import Version, InvalidVersion
from PySide6.QtCore import QThread, Signal

class UpdateChecker(QThread):
    """
    バックグラウンドでGitHubのリリース（またはタグ）を確認し、
    新しいバージョンがあればシグナルを発火するQThreadクラス。
    """
    # 更新があった場合に発火 (最新バージョン文字列, リリースページURL)
    update_available = Signal(str, str)
    
    def __init__(self, current_version: str, repo_owner: str, repo_name: str, parent=None):
        super().__init__(parent)
        self.current_version = current_version
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/tags"
        self.releases_url = f"https://github.com/{repo_owner}/{repo_name}/releases"
        
    def run(self):
        try:
            # GitHub APIを叩く (User-Agentが必須)
            req = urllib.request.Request(
                self.api_url,
                headers={'User-Agent': 'ResultsBook2DB-UpdateChecker'}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    if not data:
                        return # タグが一つもない
                        
                    # 最新のタグを取得 (GitHub APIは通常、作成日順か名前順で返す。一番最初が最新か確認が必要)
                    # より安全に、全てのタグをパースして最大のものを探す
                    latest_tag = None
                    max_version = None
                    
                    for tag_info in data:
                        tag_name = tag_info['name']
                        # 'v'プレフィックスがあれば外してパース
                        clean_tag = tag_name.lstrip('vV')
                        try:
                            v = Version(clean_tag)
                            if max_version is None or v > max_version:
                                max_version = v
                                latest_tag = tag_name
                        except InvalidVersion:
                            continue # パースできないタグは無視
                            
                    if max_version is None:
                        return
                        
                    # 現在のバージョンと比較
                    try:
                        current_v = Version(self.current_version.lstrip('vV'))
                    except InvalidVersion:
                        # 開発中などで現在のバージョンが妥当でない場合はスキップ
                        return
                        
                    if max_version > current_v:
                        self.update_available.emit(latest_tag, self.releases_url)
                        
        except (urllib.error.URLError, json.JSONDecodeError, Exception):
            # ネットワークエラー等はサイレントに無視（ユーザーの作業を邪魔しない）
            pass
