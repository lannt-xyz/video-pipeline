import pytest


@pytest.fixture
def sample_html_chapter():
    return """
    <html><body>
    <h2 class="chapter-title">Chương 1: Khởi Đầu</h2>
    <div class="chapter-content">
        Đây là nội dung chương một. Diep Thieu Duong bước vào căn phòng tối tăm,
        ánh mắt sắc bén quét qua từng góc khuất. Tiếng gió rít ngoài cửa sổ
        khiến không khí thêm phần u ám. Anh ta rút ra lá bùa từ trong tay áo,
        bắt đầu vẽ những ký hiệu huyền bí lên mặt đất.
    </div>
    </body></html>
    """


@pytest.fixture
def corrupted_html():
    return """<html><body><div class="content">x</div></body></html>"""
