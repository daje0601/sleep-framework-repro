# Sleep 논문과 ARC 구현 대조

2026년 11월 7일에 논문 v2의 식 (1)~(5), §3의 번호 없는 목적함수, 실험 표와 부록을 코드 `801e597` 및 `arc_sleep_v2.json`과 대조했다. H100 정규 학습 기록은 수정 전 코드 `cead89d`에서 얻었다. 두 버전 모두에 아래 보상식과 평균 기울기 갱신의 차이가 있다. [논문](https://arxiv.org/html/2606.03979v2) · [검토한 설정](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/configs/arc_sleep_v2.json) · [H100 정규 실행](../results/arc-h100-seed42-training-12/report.json)

## 수식과 학습 경로

| 항목 | 대조 결과 | 근거 |
| --- | --- | --- |
| 식 (1), 메모리 블록 합성 | 단계별 MLP를 차례로 적용한다. 현재 ARC 모델에서는 Llama의 각 FFN에 두 단계 메모리를 넣는다. 저랭크 잔차 블록과 토큰별 고정 라우터는 이 구현의 선택이다. | [§2.2](https://arxiv.org/html/2606.03979v2#S2.SS2) · [메모리 체인](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/memory.py#L278) |
| 식 (2), 주기적 갱신 | 코드는 주기 `[1,4]`의 경계에서 누적 기울기의 평균을 적용한다. 같은 네 기울기와 같은 고정 학습률로 SGD를 비교하면 합산 대비 ¼이다. 다만 논문은 수준·시점별 학습률을 허용하므로 각 항의 학습률을 누적 횟수로 나누면 같은 평균을 표현한다. 인쇄된 합의 양 끝 인덱스를 포함하면 C+1개이고 본문은 C단계마다 갱신한다고 설명한다. ARC 저자의 실제 정규화는 확정할 수 없다. | [§2.2](https://arxiv.org/html/2606.03979v2#S2.SS2) · [누적과 갱신](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/memory.py#L114) |
| §3.2–3.3, 메모리 전달 | 갱신 전 교사를 고정하고, 별도 학생에서 송신층의 갱신을 준비한 뒤 새 수신 expert를 학습하고 일괄 반영한다. 수정본은 기울기를 받지 못해 그대로인 expert를 모든 토큰에 추가해 한 번 더 학습한다. 이후 일반 라우팅에서 손실이 유한한지 확인한다. 이 복구 규칙과 검사만으로 교사 지식 보존은 입증되지 않는다. | [§3.3](https://arxiv.org/html/2606.03979v2#S3.SS3) · [전달 및 복구](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/consolidation.py#L235) |
| 식 (3), 보상 결합 | 의미 보상과 편집 보상을 가중합한다. 현재 가중치는 각각 0.5이며, 의미 판정에는 고정한 교사의 YES/NO 응답을 쓴다. 저자와 동일한 판정 모델인지 확인되지 않았으며 식 (4)의 차이는 합산값에도 반영된다. | [§3.3](https://arxiv.org/html/2606.03979v2#S3.SS3) · [보상](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/objectives.py#L139) · [판정기](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/engine.py#L276) |
| 식 (4), 편집거리 보상 | 논문은 **토큰** 편집거리 `z`가 임계값 `z₀`를 넘으면 보상을 0으로 둔다. 코드는 **문자열** 편집거리를 정규화하며 임계값이 없다. 논문은 임계 규칙을 명시하지만 §3.3과 ARC 부록에는 ARC용 숫자가 제시되지 않는다. | [§3.3](https://arxiv.org/html/2606.03979v2#S3.SS3) · [현재 보상 함수](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/objectives.py#L127) |
| §3.3, 증류와 최종 Knowledge Seeding 목적함수 | 앞의 GKD 식은 교사·학생 응답에서 발산을 계산한다. 뒤의 최종 KS 식은 **보상과 발산 모두 학생 응답에 대한 기대값**이다. 코드는 교사·학생 양쪽에 두 항을 모두 적용한다. 앞 식의 혼합을 유지해 발산을 계산한다는 해석은 가능하지만, 어느 식에도 교사 응답을 자기 자신과 비교해 편집 보상 1을 주는 항은 없다. 학생 쪽 발산도 교사 접두사에서 생성한 답을 재사용한다. KL 방향이나 최대화/최소화 부호의 문제가 아니다. | [§3.3 두 목적함수](https://arxiv.org/html/2606.03979v2#S3.SS3) · [샘플과 손실](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/engine.py#L318) · [손실 계산](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/objectives.py#L97) |
| §3.3, 학생 샘플 확률 | 무작위로 고른 교사 응답의 접두사에서 학생이 이어 쓰며 접두사는 손실에서 제외한다. 생성은 온도 0.8을 쓰고 보상 항의 로그확률은 원래 로짓으로 계산한다. 또한 생성 토큰을 문자열로 바꾼 뒤 다시 토큰화하고 EOS를 붙인다. 길이 한도로 끝난 응답도 끝 토큰 하나를 추가로 채점할 수 있다. | [생성](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/engine.py#L84) · [토큰 재구성](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/runtime.py#L267) · [로그확률](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/objectives.py#L77) |
| §3.4, 꿈 생성과 중요도 | 꿈 생성 시 일반 top-k 밖의 무작위 expert를 추가할 수 있다. 논문이 제시한 SFT 기울기 벡터를 순위로 만들기 위해 코드는 변형한 예시들에 대한 내부 LoRA 기울기의 L2 크기를 쓴다. 유효한 후보 중 상위 12개와 무작위 3개까지 뽑는다. | [§3.4](https://arxiv.org/html/2606.03979v2#S3.SS4) · [라우팅](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/memory.py#L170) · [중요도](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/engine.py#L218) · [선택](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/runner.py#L425) |
| 식 (5), 개선 보상과 ReST-EM | 후보마다 모델·옵티마이저·난수 상태를 같은 출발점으로 복원한다. 보류한 학습 예시 한 개에서 적응 전후의 격자 정답 점수를 비교한다. 로그는 부호 있는 차이를 담고 채택 여부는 양수일 때만 참이다. 채택된 지시문을 정규화한 JSON으로 정책 LoRA를 지도학습한다. 식 (5)의 이진 채택 조건은 구현됐다. | [§3.4](https://arxiv.org/html/2606.03979v2#S3.SS4) · [후보 검증](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/dreaming.py#L94) · [정책 업데이트](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/runner.py#L468) |

## ARC 실험과 결과 해석

- **과제를 넘어 전달되는 가중치.** 각 학습 과제와 평가 시도는 정책 모델의 새 복사본에서 시작한다. wake와 메모리 통합으로 바뀐 expert는 그 복사본에 남고, 공유 모델에는 채택된 후보의 정책 LoRA 업데이트가 저장된다. 첫 과제의 손실 하락만으로 다음 과제에 메모리가 전달됐다고 결론 내릴 수 없다. [과제 복사](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/runner.py#L269) · [체크포인트 저장](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/runner.py#L506)
- **후보 생성의 관측된 병목.** 프롬프트는 먼저 격자만 출력하라고 요구한 뒤 학습 지시문 하나만 출력하라고 요구한다. 부록 B.3은 60개에서 15개를 남긴다. 첫 H100 과제에서는 파싱 오류 40개와 스키마 오류 20개로 검증할 후보가 0개였다. 이 결과는 프롬프트 충돌과 양립하지만 인과관계는 별도 비교 실험이 필요하다. 두 번째 과제의 종료 원인은 수신 expert의 가중치 갱신 검사였다. [프롬프트](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/engine.py#L236) · [격자 지시](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/arc.py#L338) · [관측값](../results/arc-h100-seed42-training-12/episodes.jsonl)
- **데이터와 정답 접근.** SEAL 공개 자료의 커밋과 파일 해시를 고정했다. 학습 11개와 평가 8개는 복원한 목록이며 논문 저자의 정확한 과제 ID 목록은 확인되지 않았다. 학습 후보의 보상은 학습 예시 중 보류한 정답을 쓴다. 평가 데이터 로더가 파일을 읽더라도 평가 정답 배열은 적응 함수에 넘기지 않으며, 예측이 끝난 뒤 채점기에 전달된다. [부록 B.3](https://arxiv.org/html/2606.03979v2#A2.SS3) · [과제 목록](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/arc_data_manifest.json) · [평가 경로](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/runner.py#L547)
- **정확도의 분모.** 평가 8과제 × 5회는 40개 과제·시도 조합이다. 평가 목록에는 총 9개 테스트 입력이 있으며 한 시도의 모든 출력이 맞아야 그 시도가 성공한다. 코드는 누락·중복 시도를 거부한다. 지시문이 잘못되면 격자 예측을 건너뛰고 그 시도를 분모에 남긴다. 시도 성공률과 한 번이라도 푼 과제 비율은 다른 통계다. Sleep 경로는 입력당 격자 하나, 별도의 upstream SEAL 경로는 투표로 고른 격자 두 개를 허용한다. Sleep 논문 B.3에는 정답 후보 수를 확인할 만큼의 세부 설정이 없다. [채점](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/metrics.py#L32) · [평가 예측](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/runner.py#L582) · [SEAL 비교 경로](SEAL_BASELINE.md)
- **백본·튜닝 조건.** 실행 모델은 고정 리비전의 Llama-3.2-1B-Instruct다. B.3은 Llama-3.2-1B를 명시한다. 세부 리비전과 채팅 직렬화까지 같은지는 확인되지 않았다. 내부 LoRA rank 128, 정책 rank 16, 고정 학습률, 제한된 JSON 증강 방식은 현재 ARC 설정이다. 부록 B.4의 일반 학습표만으로 ARC의 내부·외부 설정을 특정할 수 없다. [부록 B.3–4](https://arxiv.org/html/2606.03979v2#A2.SS3) · [설정](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/configs/arc_sleep_v2.json) · [SEAL과의 차이](PROTOCOL_V2.md)
- **최종 보고의 검증 공백.** 평가 CLI는 체크포인트의 용도·설정·프로토콜을 확인하지만 완료 과제 수가 11인지 자체적으로 강제하지 않는다. 다중 시드 집계는 설정·과제·데이터 해시를 비교하지만 실행 코드 지문을 비교하거나 원본 40개 시도로부터 정확도를 재계산하지 않는다. 최종 보고에는 학습 로그 11행, 종료 코드, 실제 체크포인트, 평가 전체 원본 기록을 함께 확인해야 한다. 시드 42·43·44의 평가 난수열도 과제별로 일부 겹치므로 120개 시도를 독립 표본으로 해석할 수 없다. [체크포인트 검사](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/runner.py#L197) · [시도별 시드](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/runner.py#L559) · [집계](https://github.com/daje0601/sleep-framework-repro/blob/801e59749219f71ef4c46729cd1f92b90062840e/src/sleep_repro/cli.py#L122)

## 전체 논문에서 검증된 범위

| 논문 항목 | 저장소에서 확인된 범위 |
| --- | --- |
| Table 4 ARC | 학습·평가 경로가 있다. 공개된 정규 H100 실행은 학습 1/11, 정책 업데이트 0회, 평가 실행 0/40이다. 정확도는 미측정이다. |
| Tables 1–2, 수학 추론과 절제 실험 | Qwen 기반 AIME·HMMT의 학습·평가 및 평균 16회 결과를 재현한 기록이나 전용 경로가 없다. |
| Table 3, 지식 습득 | 단일·연속 문단 지식 습득 및 구성요소 제거 실험의 전용 경로와 수치가 없다. |
| Figures 3–6, §B.2, §B.5 | 연속 분류·번역, 장문 질의응답·검색, BABILong 및 동등 성능 대비 학습시간 비교의 전용 실험 경로와 수치가 없다. |

부록 A는 관련 연구를 비교한다. 그곳에 소개된 다른 방법의 수식은 Sleep의 필수 구현으로 세지 않았다. [논문 실험](https://arxiv.org/html/2606.03979v2#S4) · [관련 연구](https://arxiv.org/html/2606.03979v2#A1) · [실험 상태](EXPERIMENT_STATUS.md)

## 직접 검증한 사례

전체 로컬 테스트는 232개 통과, CUDA 전용 테스트 2개 건너뜀이다. 아래 예는 현재 파이썬 구현을 CPU에서 호출한 결과다. 실제 ARC 정확도에 미친 효과는 측정되지 않았다.

| 작은 입력 | 관측값 |
| --- | --- |
| 식 (2), 초기 가중치 1, 기울기 +1을 네 번 누적, 학습률 0.1 | 현재 가중치는 약 0.9. 같은 고정 학습률로 네 항을 합산하면 0.6이다. |
| 식 (4), `abcd`와 `abef` | 현재 편집 보상은 0.5. 각 글자가 한 토큰이고 예시 임계값이 1이라면 두 번의 편집으로 식 (4)의 보상은 0이다. 임계값 1은 설명용이다. |
| 식 (4), 실제 고정 Llama 토크나이저로 `[[1,2]]`와 `[[1,3]]` 비교 | 두 문자열은 각각 7글자·5토큰이고 어느 단위로 세어도 편집거리는 1이다. 코드의 보상은 6/7(약 0.857), 논문식은 임계값이 1 이상이면 4/5(0.8), 미만이면 0이다. **어떤 임계값으로도 일치하지 않는다.** |
| KS, 교사 샘플만 사용하고 증류·의미 보상의 가중치를 0으로 설정한 균등 로짓 모형 | 학생이 생성한 응답이 없어도 교사의 문자열을 보상 1로 채점해 학생 로짓에 0이 아닌 기울기가 생겼다. |
| EOS 없이 끝나는 단일 생성 토큰, CPU 시험용 토크나이저 | 생성 ID `[68]`에 대해 손실에 들어간 정답 ID는 `[68, 2]`였다. |
| 생성한 토큰의 재인코딩, 실제 고정 Llama 토크나이저 | 여는 대괄호 토큰 두 개를 이어서 디코딩하면 `[[`인데, 그 문자열을 다시 인코딩하면 토큰 하나다. 가능한 경로에 대한 반례이며 H100 실행에서 발생한 빈도는 기록되지 않았다. |

원문과 같은 수치 절차를 검증하려면 먼저 식 (4)의 토큰·임계 규칙, 학생 샘플에만 적용되는 보상 경로와 샘플 확률, 식 (2)의 학습률·정규화를 버전별로 명시해야 한다. 지시문 프롬프트의 영향은 고정된 학습 예시에서 유효 후보율과 실제 채택률을 각각 비교해야 한다. 원래 구현과 수정 구현의 점수는 서로 다른 실험으로 기록해야 한다.

## 재검토: 실제로 무엇이 학습됐나

첫 H100 과제 `44f52bb0`는 시연 6개 중 하나를 보상용으로 보류했다. wake 손실 5.36, 4.03, 2.22, 1.13은 남은 시연 중 **서로 다른 네 예시**를 차례로 대상으로 삼은 값이다. 동일 예시의 적응 전후 비교가 아니다. 메모리 통합에서는 기울기로 새 expert를 갱신했지만, 손실을 계산할 때마다 교사 답·접두사·학생 답을 새로 뽑는다. 학습 후에도 손실이 유한한지만 검사한다. 따라서 이 로그는 가중치 갱신의 증거지만, 같은 입력에서 교사를 더 잘 모방하거나 문제를 더 잘 풀었다는 측정값은 아니다. [wake 예시 선택](../src/sleep_repro/runner.py#L277) · [응답 생성](../src/sleep_repro/engine.py#L318) · [사후 검사](../src/sleep_repro/consolidation.py#L170) · [실행 손실](../results/arc-h100-seed42-training-12/episodes.jsonl)

실행 보고서에 기록된 SHA-256과 원본 `training.jsonl`이 일치하는지 확인한 뒤, 탈락한 60개 응답의 텍스트를 다시 검사했다. **응답 전체**를 원래 파서에 넣으면 통과한 것은 여전히 0개다. 각 여는 중괄호에서 시작하는 JSON 객체만 잘라 같은 파서로 검사하면 18개 응답에 유효한 지시문이 들어 있었다. 그중 15개는 다른 내용 뒤에 유효한 지시문으로 끝났으며, 그 15개 중 13개는 프롬프트 예시와 같은 기본 설정이었다. 따라서 모델이 지시문 형식을 전혀 생성하지 못했다는 설명은 맞지 않는다. 반대로 이 지시문이 유용했다는 증거도 없다. 실제로 적응·채점한 후보는 0개이고 소급하여 채택하지 않았다. [집계 방법·후보 인덱스](../results/arc-h100-seed42-training-12/directive-format-audit.json) · [원래 파서](../src/sleep_repro/arc.py#L267)

## 재검토: SEAL 원본을 따라가면 달라지는 ARC 조건

[Sleep 부록 B.3](https://arxiv.org/html/2606.03979v2#A2.SS3)은 학습 시 60개 중 45개를 버리고, 평가 시 과제당 5개의 꿈을 독립 적용한 뒤 성공한 비율을 낸다고 명시한다. 아래는 **공개 SEAL과 우리 코드 사이에서 확인한 차이**다. Sleep 저자가 공개하지 않은 구현 세부까지 SEAL과 같았다고 가정하지 않았다.

| 비교 항목 | 공개 SEAL | 현재 ARC 재구현 |
| --- | --- | --- |
| 한 시도로 세는 대상 | [생성 루프](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot/self-edit.py#L426)는 JSON 해석 오류를 건너뛰고 설정을 요구된 수만큼 모은다. | [평가 루프](../src/sleep_repro/runner.py#L557)는 원시 생성 5회를 센다. 형식 오류이면 그 시도는 격자를 예측하지 않고 실패한다. 매 시도마다 모델 복사본에서 wake와 통합도 다시 실행한다. B.3은 형식 오류 처리나 5회 간 통합 상태 공유 여부를 명시하지 않는다. |
| 지시문을 만들 때 보이는 것 | [프롬프트](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot/self-edit.py#L343)는 시연의 입력·출력만 보여 준다. | [지시문 프롬프트](../src/sleep_repro/engine.py#L236)에 시연과 **첫 테스트 입력**이 들어간다. 테스트 입력이 두 개인 `b1fc8b8e`에서는 첫 입력을 보고 만든 지시문을 양쪽에 적용한다. 테스트 정답은 제공하지 않는다. |
| 무엇을 맞히면 후보를 학습하는가 | [논문 §4.1](https://arxiv.org/html/2506.10943#S4.SS1)과 [실행 순서](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot/README.md#L11-L49)는 같은 정책에서 후보를 모으고 **훈련 분할**의 공식 질의 정답으로 채점한 뒤, 성공 후보를 모아 외부 정책을 학습한다. | [학습 루프](../src/sleep_repro/runner.py#L411)는 훈련 과제마다 마지막 시연을 가려서 채점하고, 이전보다 개선된 후보가 있으면 **다음 과제 전에** 정책을 갱신한다. `5614dbcf`, `6150a2bd`는 가리고 나면 한 쌍만 남아, 다른 시연 없이 그 한 쌍으로 내부 학습한다. |
| 수치의 출처 | [SEAL Table 1](https://arxiv.org/html/2506.10943#S4.SS1)은 정책 학습 전 self-edit 20%, SEAL 72.5%, 수작업 최적 설정(Oracle TTT) 100%를 구분한다. [공개 결과 파일](https://github.com/Continual-Intelligence/SEAL/blob/6d9c9f9ee392c6cc618e771f399d436d190f6ca4/few-shot/final_results.json)은 한 인덱스 아래 오답 8개뿐이다. | [Sleep Table 4](https://arxiv.org/html/2606.03979v2#S4.SS1)의 TTT 10%를 위의 어느 조건과도 자동으로 동일시할 수 없다. 공개 파일은 8개 과제 ID를 복원하는 근거이지, SEAL 72.5% 또는 Sleep 80%를 산출한 원시 40회 로그가 아니다. |

[Sleep 부록 B 첫 문단](https://arxiv.org/html/2606.03979v2#A2)은 폭 64의 추가 MLP 5개와 활성 파라미터 수 유지를 설명하면서 3B·8B 백본을 명시한다. 이를 ARC 1B에 그대로 적용해야 하는지는 B.3에서 알 수 없다. 우리의 [라우터](../src/sleep_repro/memory.py#L203)는 토큰별로 expert를 선택하지만, 배치에서 선택된 각 expert를 전체 토큰에 실행한 뒤 마스크를 씌운다. CPU에서 두 토큰이 서로 다른 top-1 expert를 선택하게 했을 때 두 expert 모두 두 토큰을 입력받았다. 따라서 [활성 파라미터 수](../src/sleep_repro/model.py#L123)나 top-1 설정은 실제 GPU 연산량 또는 논문 §B.5의 효율을 측정한 값이 아니다.

후속 검증의 순서는 (1) 평가 정답을 보지 않고 식 (4)의 토큰·임계 규칙과 KS 샘플 분포를 고정하고, (2) **학습 분할**에서 프롬프트만 바꿔 전체 지시문 통과율·실제 후보 채택률을 따로 측정하며, (3) 평가 분모·실패 규칙·입력당 격자 수를 등록한 뒤 고정된 체크포인트를 평가하는 것이다. 기존 실행은 원래 규칙으로 끝까지 기록하고, 변경한 조건은 별도 버전으로 비교한다.
