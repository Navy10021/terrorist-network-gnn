# CI/CD 실패 시 빠른 수정 가이드

## 🚨 실패 원인별 해결책

### 1. Import 오류

**증상**:
```
ModuleNotFoundError: No module named 'torch'
ImportError: cannot import name 'xxx'
```

**해결**:
```bash
# 의존성 확인
cat requirements.txt

# CI/CD 워크플로우 확인
cat .github/workflows/ci.yml

# 필요 시 의존성 추가
echo "missing-package>=1.0.0" >> requirements.txt
```

---

### 2. 테스트 실패

**증상**:
```
FAILED tests/test_xxx.py::TestClass::test_method
AssertionError: ...
```

**해결**:
```bash
# 로컬에서 실패한 테스트만 실행
pytest tests/test_xxx.py::TestClass::test_method -v

# 문제 수정
# 1. 테스트 로직 확인
# 2. 모듈 코드 확인
# 3. 픽스처 확인

# 재테스트
pytest tests/test_xxx.py -v
```

---

### 3. 커버리지 미달

**증상**:
```
FAIL Required test coverage of 20% not reached. Total coverage: 15%
```

**해결**:
```bash
# 옵션 A: 임시로 임계값 낮추기
# pyproject.toml 수정
--cov-fail-under=15

# 옵션 B: 테스트 추가 (권장)
# 커버리지 낮은 모듈 확인 후 테스트 작성
```

---

### 4. Flake8/mypy 경고

**증상**:
```
src/xxx.py:123: E501 line too long
```

**해결**:
```bash
# Black 포맷팅
black src/ tests/

# isort
isort --profile black src/ tests/

# Flake8는 continue-on-error: true이므로 merge 가능
```

---

### 5. PyTorch Geometric 설치 실패

**증상**:
```
ERROR: Could not find a version that satisfies the requirement torch-geometric
```

**해결**:
```yaml
# .github/workflows/ci.yml 수정
- name: Install PyTorch Geometric
  run: |
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    pip install torch-geometric
```

---

## 📝 표준 수정 프로세스

### 1. 에러 로그 분석
```bash
# GitHub Actions에서 실패한 job 클릭
# 에러 메시지 복사
# 원인 파악
```

### 2. 로컬에서 재현
```bash
# 같은 환경 구성
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
pip install pytest pytest-cov

# 테스트 실행
pytest tests/ -v
```

### 3. 문제 수정
```bash
# 코드 수정
# 테스트 추가/수정
# 포맷팅 적용
```

### 4. 재커밋 및 푸시
```bash
git add .
git commit -m "fix: CI/CD 실패 수정 - [구체적 설명]"
git push
```

### 5. CI/CD 재실행 확인
```bash
# GitHub Actions에서 자동 재실행
# 모든 체크 통과 확인
```

---

## 🎯 Merge 체크리스트

Pull Request를 merge하기 전에 확인:

### 필수 (모두 ✅ 여야 함)
- [ ] 모든 CI/CD 테스트 통과
- [ ] 코드 품질 검사 통과
- [ ] 테스트 커버리지 20% 이상
- [ ] 빌드 성공
- [ ] 충돌(conflict) 없음

### 권장
- [ ] 코드 리뷰 받음
- [ ] CHANGELOG 업데이트
- [ ] 문서 업데이트
- [ ] Breaking changes 문서화

---

## 🚀 긴급 상황

### Hotfix가 필요한 경우

**프로덕션 긴급 버그**:
```bash
# 1. hotfix 브랜치 생성
git checkout -b hotfix/critical-bug

# 2. 최소한의 수정
# 3. 테스트 확인
# 4. 직접 main에 merge (리뷰 생략 가능)

# 단, CI/CD는 반드시 통과해야 함!
```

---

## 📞 도움 요청

### 해결이 안 되면
1. **GitHub Issues** - 버그 리포트
2. **GitHub Discussions** - 질문
3. **팀원에게 문의** - 코드 리뷰 요청

---

## 🔄 지속적 개선

### CI/CD 실패율 줄이기
- 로컬에서 CI/CD와 동일한 검사 실행
- pre-commit hooks 사용
- 테스트 작성 습관화
- 의존성 버전 고정

### 모니터링
- CI/CD 실패 패턴 분석
- 자주 실패하는 테스트 개선
- 빌드 시간 최적화

---

**원칙**: "실패한 CI/CD는 고치고 나서 merge" 🎯
