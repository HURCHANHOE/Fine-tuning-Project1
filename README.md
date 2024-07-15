# Fine-tuning-Project1
    PAUST 에서 개발한 한국어 기반의 encoder-decoder 모델의 Summarization 타입의 Framework

## 폴더별 설명
    data : 학습데이터
    interface : train, service 파이프라인이 반드시 구현해야 하는 메서드 정의
    nlp
        common : 공통 코드들의 집합
        summarizationnlp : corpus 분리, encoding, trainer 생성하는 코드 정리
    repository : 학습에 필요한 config 세팅
    train : 학습 소스코드
    service : 추론 소스코드

## 하이퍼 파라미터
    gradient_accumulation_steps : Number of updates steps to accumulate the gradients for, before performing a backward/update pass. (int)
    max_seq_length : The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. (int)
    split_ratio : Train-Validation Split ratio"
    epochs : Total number of training epochs to perform. (int)​"
    learning_rate : The initial learning rate for the optimizer. (float)​
    early_stopping_patience : Use with metric_for_best_model to stop training when the specified metric worsens for early_stopping_patience evaluation calls.(int, scope : 3~5)
## 라이센스
PAUST에서 만든 pko-t5는 MIT license 하에 공개되어 있습니다.

## References
https://github.com/paust-team/pko-t5
https://github.com/huggingface/transformers
