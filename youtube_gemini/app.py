import streamlit as st
import time
import google.generativeai as genai
import os
from pytubefix import YouTube
import base64
import ssl
import logging
from google.api_core.exceptions import ServiceUnavailable
from typing import Tuple, Optional

# SSL 인증서 검증 비활성화 (환경에 따라 필요)
ssl._create_default_https_context = ssl._create_unverified_context

# 로그 설정: 파일과 콘솔 동시 출력, UTF-8 인코딩
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# API 키 환경변수에서 읽기
API_KEY: Optional[str] = os.getenv("GENAI_API_KEY")
if not API_KEY:
    st.error("환경변수 'GENAI_API_KEY'에 API 키를 설정해 주세요.")
    logging.error("환경변수 'GENAI_API_KEY'가 설정되지 않음, 앱 종료")
    st.stop()

# Gemini API 키 구성
genai.configure(api_key=API_KEY)

def download_youtube(url: str) -> str:
    """
    유튜브 URL로부터 최고 해상도 영상 다운로드 후 로컬 파일 경로 반환.

    Args:
        url (str): 유튜브 영상 URL

    Returns:
        str: 다운로드한 영상 파일 경로

    Raises:
        Exception: 다운로드 실패 시 예외 발생 가능
    """
    logging.info(f"유튜브 영상 다운로드 시도: {url}")
    try:
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        file_path = stream.download(output_path="./videos")
        logging.info(f"영상 다운로드 성공: {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"유튜브 영상 다운로드 실패: {e}")
        raise

def wait_for_file_active(uploaded_file, timeout: int = 300, interval: int = 1):
    """
    업로드한 파일이 'ACTIVE' 상태가 될 때까지 대기.

    Args:
        uploaded_file: Gemini API에서 반환된 업로드 파일 객체
        timeout (int): 최대 대기 시간(초)
        interval (int): 상태 체크 주기(초)

    Returns:
        uploaded_file: 활성화된 파일 객체

    Raises:
        TimeoutError: 지정 시간 내에 ACTIVE 상태가 안 될 경우
    """
    logging.info(f"파일 활성화 대기 시작: {uploaded_file.name}")
    start_time = time.time()
    while uploaded_file.state != 2:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            logging.error("파일 활성화 대기 시간 초과")
            raise TimeoutError("파일이 ACTIVE 상태가 되지 않았습니다.")
        time.sleep(interval)
        uploaded_file = genai.get_file(uploaded_file.name)
    logging.info("파일 활성화 완료")
    return uploaded_file

def delete_file(file_path: str, uploaded_file) -> None:
    """
    로컬 파일과 업로드된 파일 객체 삭제.

    Args:
        file_path (str): 로컬 파일 경로
        uploaded_file: Gemini API 업로드 파일 객체
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"로컬 파일 삭제 완료: {file_path}")
    except Exception as e:
        logging.warning(f"로컬 파일 삭제 실패: {e}")

    try:
        uploaded_file.delete()
        logging.info(f"업로드 파일 삭제 완료: {uploaded_file.name}")
    except Exception as e:
        logging.warning(f"업로드 파일 삭제 실패: {e}")

def recog_video(
    prompt: str,
    url: str,
    model,
    max_retries: int = 5,
    retry_delay: int = 10
) -> Tuple[str, object, str]:
    """
    유튜브 영상 다운로드 → 업로드 → Gemini 모델 분석 후 결과 반환.
    재시도 로직 포함.

    Args:
        prompt (str): AI 모델에 전달할 질문/명령어
        url (str): 유튜브 영상 URL
        model: genai.GenerativeModel 인스턴스
        max_retries (int): 최대 재시도 횟수
        retry_delay (int): 재시도 전 대기 시간(초)

    Returns:
        Tuple[str, object, str]: (로컬 파일 경로, 업로드 파일 객체, AI 생성 답변)

    Raises:
        RuntimeError: 모델 과부하 등으로 재시도 실패 시 예외 발생
    """
    logging.info("영상 AI 분석 시작")
    file_path = download_youtube(url)
    uploaded_file = genai.upload_file(path=file_path)
    uploaded_file = wait_for_file_active(uploaded_file)
    contents = [prompt, uploaded_file]

    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"AI 모델 요청 시도 {attempt}/{max_retries}")
            responses = model.generate_content(
                contents,
                stream=True,
                request_options={"timeout": 120}
            )
            answer = ""
            for response in responses:
                answer += response.text
            logging.info("AI 답변 수신 완료")
            return file_path, uploaded_file, answer
        except ServiceUnavailable as e:
            logging.warning(f"모델 과부하 에러: {e}")
            if attempt < max_retries:
                st.warning(f"모델 과부하로 재시도 중... ({attempt}/{max_retries})")
                time.sleep(retry_delay)
            else:
                delete_file(file_path, uploaded_file)
                logging.error("재시도 실패, 모델 과부하 상태")
                raise RuntimeError("Gemini 모델이 과부하 상태입니다. 나중에 다시 시도해주세요.") from e
        except Exception as e:
            logging.error(f"AI 모델 요청 중 알 수 없는 오류 발생: {e}")
            delete_file(file_path, uploaded_file)
            raise

def set_background_image(image_path: str) -> None:
    """
    스트림릿 앱 배경 이미지를 지정하는 함수.

    Args:
        image_path (str): 배경 이미지 파일 경로
    """
    logging.info(f"배경 이미지 설정: {image_path}")
    try:
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
            encoded = base64.b64encode(img_bytes).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception as e:
        logging.warning(f"배경 이미지 설정 실패: {e}")

# 배경 이미지 경로 지정 (본인 환경에 맞게 변경)
background_image_path = "/Users/jini/Downloads/jini/st_coding/gemini_youtube/background.jpg"
set_background_image(background_image_path)

# Streamlit UI 시작
st.title("유튜브 영상 AI 분석")

url = st.text_input(
    "유튜브 영상 URL을 입력하세요:",
    value="https://www.youtube.com/watch?v=-psgagqWoIo"
)

prompt_default = """
- 어떤 사람이 주인공인지 인상착의를 말해주세요.
- 보고 느낀 점을 한 문장으로 말해주세요.
- 가장 인상적인 장면을 한 문장으로 말해주세요.
"""

prompt = st.text_area(
    "질문 내용을 입력하세요:",
    value=prompt_default,
    height=250
)

if st.button("분석 시작"):
    model_name = os.getenv("GENAI_MODEL_NAME", "gemini-2.0-flash")
    model = genai.GenerativeModel(model_name=model_name)
    try:
        with st.spinner("동영상 다운로드 및 업로드 중... 잠시만 기다려주세요."):
            file_path, uploaded_file, answer = recog_video(prompt, url, model)

        st.video(file_path)

        # 답변 박스 스타일 CSS
        box_style = """
        <style>
        .box {
            border: 2px solid #4a90e2;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            background-color: #f0f8ff;
            box-shadow: 2px 2px 5px rgba(74,144,226,0.3);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .box-title {
            font-weight: bold;
            font-size: 20px;
            color: #004080;
            margin-bottom: 10px;
        }
        .box-content {
            white-space: pre-wrap;
            font-size: 16px;
            color: #222222;
        }
        </style>
        """
        st.markdown(box_style, unsafe_allow_html=True)

        # AI 답변 출력
        st.markdown(f"""
        <div class="box">
            <div class="box-title">답변:</div>
            <div class="box-content">{answer}</div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        logging.error(f"분석 중 오류 발생: {e}")
        st.error(f"에러 발생: {str(e)}")

    finally:
        try:
            delete_file(file_path, uploaded_file)
            logging.info("임시 파일 삭제 완료")
            st.success("분석 완료")
        except Exception as e:
            logging.warning(f"임시 파일 삭제 중 오류: {e}")
