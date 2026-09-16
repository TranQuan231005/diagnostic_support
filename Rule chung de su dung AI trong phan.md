Rule chung de su dung AI trong phan tich, thiet ke ,coding, testing va documentation



##### 6 Nguyen tac cot loi

1\. Hieu truoc khi code

2\. len ke hoach

3\. prompt nho, thay doi nho

4\. chay \& xac nhan moi thay doi

5\. AI output is untrusted

6\. Giai thich code





##### Workflow chuan



Define -> Design -> Break Down -> Build MVP -> Implement -> Test -> Refactor -> commit -> document



1. ##### Define \& Design



* **Rule:** Chua viet code den khi bai toan va cau truc he thong du ro
* **Define:** Project name, problem, target users, input/output, Core features, Technology, Evaluation critetia



* **Design:** System architecture, Modules, Folder Structure, Data Flow, Interfaces/APIs, Milestones, Technical risks



##### 2\. Build MVP -> implement module by module



MPV truoc, Mot module/1 lan, Mot thay doi/lan



##### 3\. Debug \& test co he thong.

* **Debug prompt:** Expected output, Actual behiavior, Error message, Relevant code, Environment, Recent changes (Yeu cau AI:phan tich Nguyen nhan -> Xep hang gia thuyet -> cach kiem tra -> ban sua nho nhat)
* kich ban test tung features.



* **minimum tests:** Normal cases, Invalid case, Edge cases, Failure cases, Integration tests (AI/ML: split du lieu, baseline, metrics, leakage, check, error analysis



##### 4\. Prompt template chuan

* plan , implement, review, debug



##### 5\. Ai Usage \& Documentation

* Readme.md: 1. Problem overview, project statement, main features, system architecute, folder structure, installation/run/test, Example input/output, results,  limitation, future improvent (link file vao trong readme)
* Ai\_Usage.md: Tools used, Main purpose, Student contribution, Important AI-generated component, ...



##### 6\. Nhung hanh vi khong duoc phep

* Ai tao toanf bo roi nop
* Xoa test de lamf cho pipline passed
* Dua Api ket/password/du lieu nhay cam vao prompt
* khang dinh chuc nang hoat dong khi chua co ban chung
* dung code ma khong chay thu hoac khong hieu
* chinh sua ket qua danh gia theo ky vong
* commit.env token hoac secrets
* khong khai bao viec su dung AI



##### 7\. Definition of Done

* Dung yeu cau
* Code chay
* test pass
* no secrets
* code reviewd
* Giai thich duoc 
* doc updated
* Commited



