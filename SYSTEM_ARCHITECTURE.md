# STEADYVIEW 시스템 구성도 시각화 명세

완성된 다이어그램 파일:

- SVG 원본: `SYSTEM_ARCHITECTURE_DIAGRAM.svg`
- PNG 이미지: `SYSTEM_ARCHITECTURE_DIAGRAM.png`
- 브라우저 미리보기: `SYSTEM_ARCHITECTURE_DIAGRAM.html`

아래 내용을 바탕으로 예시 이미지와 같은 한 장짜리 시스템 구성도를 제작한다.  
스타일은 흰 배경, 검은 외곽선, 둥근 사각형 모듈, 간단한 아이콘, 색상 화살표를 사용하는 발표용 아키텍처 다이어그램이다.

## 1. 전체 배치

- 캔버스 상단 왼쪽에 굵은 테두리 박스로 제목을 배치한다.
  - 제목: `시스템 구성도`
- 전체 캔버스는 16:9 비율의 가로형 구성으로 만든다.
- 모든 큰 영역이 캔버스 안에 완전히 보이도록 한다. 오른쪽 `External Runtime & Storage` 영역이 잘리면 안 된다.
- 가로 공간이 부족하면 `External Runtime & Storage`는 오른쪽 끝이 아니라 `Server`와 `Model` 아래쪽에 가로 박스로 배치한다.
- 큰 영역은 왼쪽에서 오른쪽으로 다음 순서로 배치한다.
  - `User`
  - `Front`
  - `FastAPI Server`
  - `AI / Video Processing Model`
  - `External Runtime & Storage`
- 각 큰 영역은 검은색 둥근 사각형 테두리로 묶는다.
- 각 영역 상단에는 대표 기술명을 함께 표시한다.
  - Front: `HTML / CSS / JavaScript`
  - Server: `FastAPI`
  - Model: `PyTorch / OpenCV`
  - External Runtime: `FFmpeg / Real-ESRGAN / CUDA GPU`

## 2. 영역별 구성 요소

### User 영역

왼쪽 가장자리에 사용자 아이콘을 배치한다.

표시 요소:

- 사람 아이콘
- 라벨: `User`
- 설명: `영상 업로드 및 복원 결과 확인`

User에서 Front로 향하는 검은색 화살표를 연결한다.

화살표 라벨:

- `영상 파일 업로드`
- `옵션 선택`

Front에서 User로 돌아오는 검은색 화살표도 연결한다.

화살표 라벨:

- `미리보기 / 복원 결과`
- `결과 영상 다운로드`

### Front 영역

User 오른쪽에 스마트폰 또는 브라우저 화면 형태의 박스를 배치한다.

내부 구성 요소:

- `index.html`
  - 화면 구조
- `styles.css`
  - UI 스타일
- `app.js`
  - 업로드, 미리보기, 처리 요청, 비교 재생, 다운로드 제어

Front 내부에는 다음 UI 기능을 작은 아이콘과 함께 세로로 표시한다.

- `파일 업로드`
- `복원 옵션 선택`
- `원본 영상 미리보기`
- `처리 진행률 표시`
- `원본/결과 비교 재생`
- `MP4 다운로드`

Front에서 Server로 향하는 파란색 화살표를 3개 연결한다.

화살표:

- `GET /`
  - 정적 페이지 요청
- `POST /api/preview`
  - 업로드 영상 미리보기 요청
- `POST /api/process`
  - 선택 옵션 + 영상 처리 요청

Server에서 Front로 돌아오는 파란색 화살표를 3개 연결한다.

화살표:

- `index.html + static assets`
- `preview MP4 bytes`
- `processed MP4 bytes`

### FastAPI Server 영역

중앙에 가장 큰 서버 박스를 배치한다.

상단 라벨:

- `FastAPI Server`
- 파일명: `app.py`

내부 구성 요소를 위에서 아래로 배치한다.

1. `Static File Serving`
   - `/`
   - `/static/*`
2. `API Routes`
   - `GET /api/health`
   - `GET /api/options`
   - `GET /api/runtime`
   - `POST /api/preview`
   - `POST /api/process`
3. `Validation`
   - 허용 확장자: `.mp4`, `.mov`, `.avi`
   - 허용 옵션 검증
4. `video_processor.py`
   - 브라우저 재생 가능 MP4 변환
   - 임시 파일 생성
   - 선택된 파이프라인 실행
   - 결과 bytes 반환

Server 내부의 `app.py`에서 `video_processor.py`로 검은색 화살표를 연결한다.

화살표 라벨:

- `video bytes + option 전달`

Server에서 Model 영역으로 향하는 주황색 화살표를 연결한다.

화살표 라벨:

- `복원 파이프라인 실행`

Model에서 Server로 돌아오는 주황색 화살표를 연결한다.

화살표 라벨:

- `처리 완료 MP4 반환`

### AI / Video Processing Model 영역

Server 오른쪽에 모델 처리 영역을 배치한다.

상단 라벨:

- `AI / Video Processing Model`
- `backend/pipelines`

내부에 3개의 처리 파이프라인 박스를 세로로 배치한다.

#### 1. Stabilization

라벨:

- `Stabilization`
- `backend/pipelines/stabilization`

설명:

- 특징점 추적
- 카메라 경로 계산
- 경로 스무딩
- 안정화 영상 렌더링

아이콘:

- 흔들리는 카메라 또는 그래프 아이콘

#### 2. Upright Correction

라벨:

- `Upright Correction + Stabilization`
- `backend/pipelines/upright_stabilization`

설명:

- Upright 모델 체크포인트 사용
- 기울어진 영상의 수평 보정
- 안정화 결과 생성

아이콘:

- 수평선 보정 아이콘 또는 AI 모델 아이콘

#### 3. Superresolution

라벨:

- `Superresolution`
- `backend/pipelines/superresolution`

설명:

- Real-ESRGAN 비디오 추론 호출
- 해상도 향상
- 최종 MP4 생성

아이콘:

- 확대 렌즈 또는 고해상도 이미지 아이콘

Model 영역 안에서 처리 순서를 아래처럼 표현한다.

- `Stabilization`과 `Upright Correction`은 항상 순차 실행되는 관계가 아니다.
- `Upright Correction`이 선택되면 `run_upright_stabilization()`이 실행되며, 이 파이프라인 안에 안정화 처리가 포함된다.
- `Upright Correction`이 선택되지 않고 `Stabilization`만 선택된 경우에만 `run_stabilization()`이 실행된다.
- `Superresolution`이 선택된 경우 후처리 단계로 실행된다.
- `Superresolution`만 선택한 경우 바로 Real-ESRGAN으로 전달된다.

이를 표현하는 내부 화살표:

- `Option Branch` -> `Stabilization`
- `Option Branch` -> `Upright Correction + Stabilization`
- `Option Branch` -> `Superresolution only`
- `Stabilization` -> `Superresolution 선택 시`
- `Upright Correction + Stabilization` -> `Superresolution 선택 시`
- 각 파이프라인 -> `MP4 Output`

주의:

- `Stabilization` -> `Upright Correction`처럼 항상 이어지는 직렬 화살표로 그리지 않는다.
- `Stabilization`과 `Upright Correction + Stabilization`은 선택 옵션에 따라 갈라지는 대체 경로로 표현한다.
- `Superresolution`만 선택되는 경로도 별도로 표시한다.

### External Runtime & Storage 영역

가장 오른쪽 또는 Model 영역 하단에 외부 실행 환경 박스를 배치한다.

내부 구성 요소:

1. `FFmpeg / FFprobe`
   - AVI 또는 비호환 MP4를 브라우저 재생 가능 MP4로 변환
   - H.264 / AAC / faststart 처리
2. `TemporaryDirectory`
   - 업로드 bytes를 임시 input 파일로 저장
   - 파이프라인별 output 파일 생성
   - 요청 종료 후 자동 삭제
3. `CUDA GPU / CPU`
   - PyTorch 실행 장치
   - GPU 사용 가능 시 CUDA 사용
4. `Upright .pth Model`
   - 기본 경로: `backend/pipelines/upright_stabilization/models/*.pth`
   - 환경변수: `STEADYVIEW_UPRIGHT_MODEL_PATH`
5. `Real-ESRGAN`
   - 기본 경로: `C:\Users\korea\Desktop\Real-ESRGAN\Real-ESRGAN_V2`
   - 환경변수: `STEADYVIEW_REAL_ESRGAN_DIR`

Server 또는 Model에서 External Runtime으로 향하는 점선 화살표를 연결한다.

화살표 라벨:

- `파일 변환`
- `임시 파일 입출력`
- `GPU 연산`
- `외부 추론 스크립트 호출`

External Runtime 연결 규칙:

- `POST /api/preview` 흐름은 `Server -> FFmpeg / FFprobe -> Server`로 표현한다.
- `POST /api/preview`는 AI Model 파이프라인을 거치지 않는다.
- `POST /api/process` 흐름만 AI Model 파이프라인으로 연결한다.
- `TemporaryDirectory`는 `video_processor.py`와 연결한다.
- `CUDA GPU / CPU`는 `Stabilization`, `Upright Correction`, `Superresolution`과 연결한다.
- `Real-ESRGAN`은 `Superresolution`과만 연결한다.
- `Upright .pth Model`은 `Upright Correction + Stabilization`과만 연결한다.

## 3. 주요 데이터 흐름

다이어그램에는 아래 흐름이 한눈에 보이도록 화살표 번호를 붙인다.

1. `User -> Front`
   - 사용자가 MP4/MOV/AVI 영상을 업로드하고 복원 옵션을 선택한다.
2. `Front -> Server`
   - `POST /api/preview`로 미리보기용 영상을 요청한다.
3. `Server -> FFmpeg`
   - 브라우저 재생이 어려운 형식은 MP4로 변환한다.
4. `Server -> Front`
   - 미리보기 MP4 bytes를 반환한다.
5. `Front -> Server`
   - `POST /api/process`로 선택 옵션과 영상 파일을 전송한다.
6. `Server -> Model`
   - `video_processor.py`가 옵션에 맞는 파이프라인을 실행한다.
   - 단, 미리보기 요청인 `/api/preview`는 Model을 거치지 않는다.
7. `Model -> External Runtime`
   - PyTorch, OpenCV, CUDA GPU, Real-ESRGAN, Upright 모델을 사용한다.
8. `Model -> Server`
   - 처리된 MP4 파일 bytes를 반환한다.
9. `Server -> Front`
   - `video/mp4` 응답과 다운로드 파일명을 반환한다.
10. `Front -> User`
    - 원본/결과 비교 재생 및 결과 다운로드를 제공한다.

## 4. 화살표 색상 규칙

범례를 다이어그램 오른쪽 아래에 넣는다.

- 검은색 실선: `사용자 조작 / 화면 결과`
- 파란색 실선: `웹 클라이언트 ↔ FastAPI 서버 HTTP 통신`
- 주황색 실선: `서버 ↔ AI 처리 파이프라인`
- 초록색 실선: `AI 파이프라인 내부 처리 순서`
- 회색 점선: `외부 런타임 / 파일 시스템 / GPU 의존성`

## 5. 최종 다이어그램에 반드시 포함할 텍스트

다음 텍스트는 다이어그램에 직접 표시한다.

- `STEADYVIEW`
- `HTML / CSS / JavaScript Front`
- `FastAPI Server`
- `app.py`
- `video_processor.py`
- `POST /api/preview`
- `POST /api/process`
- `Stabilization`
- `Upright Correction`
- `Superresolution`
- `FFmpeg / FFprobe`
- `Real-ESRGAN`
- `PyTorch`
- `CUDA GPU / CPU`
- `TemporaryDirectory`
- `MP4 Preview`
- `Processed MP4 Download`

## 6. 권장 시각 스타일

- 예시 이미지처럼 손으로 그린 듯한 단순한 아키텍처 다이어그램 톤으로 구성한다.
- 각 기술 로고는 가능하면 간단한 아이콘 또는 텍스트 로고로 표현한다.
  - FastAPI
  - PyTorch
  - Docker
  - FFmpeg
  - Python
  - GPU
- 너무 복잡한 코드 파일 목록은 넣지 말고, 핵심 컴포넌트 중심으로 요약한다.
- `Front`, `Server`, `Model`, `External Runtime`의 경계가 명확히 보이게 한다.
- 처리 흐름은 왼쪽에서 오른쪽으로 읽히게 한다.
- 결과 반환 흐름은 오른쪽에서 왼쪽으로 돌아오는 화살표로 표현한다.
- 화살표 라벨은 박스 내부 텍스트와 겹치지 않게 충분히 떨어뜨린다.
- `GET /api/health`, `GET /api/runtime`, `GET /api/options`는 `Static File Serving` 박스가 아니라 `API Routes` 박스에 넣는다.
- Model 영역에는 선택 분기 다이아몬드 또는 `Option Branch` 박스를 넣어 `Stabilization`, `Upright Correction + Stabilization`, `Superresolution only`가 선택 경로임을 보여준다.

## 7. 한 문장 요약

STEADYVIEW는 브라우저에서 업로드한 영상을 FastAPI 서버로 전송하고, `video_processor.py`가 선택 옵션에 따라 Stabilization, Upright Correction, Superresolution 파이프라인과 FFmpeg/Real-ESRGAN/GPU 런타임을 사용해 처리한 뒤, 브라우저에 미리보기 및 다운로드 가능한 MP4 결과를 반환하는 영상 복원 시스템이다.
