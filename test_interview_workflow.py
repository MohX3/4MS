"""
Test and Evaluation Script for 4MSHire AI Interview Stage
==========================================================
This script tests the interview workflow system with:
- 20 different job descriptions across various positions
- 20 candidates with different resume qualities (weak, average, strong, exceptional)
- Comprehensive evaluation of:
  1. Interview flow consistency
  2. Question relevance to job description (using semantic similarity)
  3. Question difficulty consistency
  4. Recommendation variation based on candidate answers
  5. Report structure consistency

All outputs are saved to test_outputs/ for inclusion in the project report.
Includes visualization generation for metrics (graphs, histograms, heatmaps).
"""

import os
import sys
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import traceback
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model

# For semantic similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files

# Project imports
from src.dynamic_workflow import (
    AgentState,
    build_workflow,
    interviewer_prompt,
    evaluator_prompt,
    report_writer_prompt,
    get_current_date_4ms
)

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "test_outputs"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Initialize LLM for candidate response simulation
llm = init_chat_model("google_genai:gemini-2.5-flash-lite", temperature=0.3)

# Initialize sentence transformer for semantic similarity
print("Loading sentence transformer model for semantic similarity...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")

# ============================================================================
# JOB DESCRIPTIONS (20 Different Positions)
# ============================================================================

JOB_DESCRIPTIONS = [
    {
        "id": "JD001",
        "title": "Senior Python Developer",
        "company": "TechCorp",
        "requirements": """
        Position: Senior Python Developer
        Location: Remote
        Employment Type: Full-time

        Key Responsibilities:
        - Design and develop scalable Python applications
        - Lead code reviews and mentor junior developers
        - Implement RESTful APIs using Django/Flask
        - Optimize database queries and application performance

        Must-Have Requirements:
        - 5+ years of Python experience
        - Strong knowledge of Django or Flask frameworks
        - Experience with PostgreSQL or MySQL
        - Proficiency in Git and CI/CD pipelines

        Nice-to-Have:
        - Experience with Docker and Kubernetes
        - Knowledge of cloud services (AWS/GCP)
        - Familiarity with microservices architecture
        """,
        "key_skills": ["Python", "Django", "Flask", "PostgreSQL", "MySQL", "Git", "CI/CD", "REST API", "Docker", "Kubernetes", "AWS", "microservices"]
    },
    {
        "id": "JD002",
        "title": "Frontend React Developer",
        "company": "WebSolutions Inc.",
        "requirements": """
        Position: Frontend React Developer
        Location: Hybrid
        Employment Type: Full-time

        Key Responsibilities:
        - Build responsive and interactive user interfaces using React
        - Collaborate with UX designers to implement pixel-perfect designs
        - Manage application state using Redux or Context API
        - Write unit and integration tests

        Must-Have Requirements:
        - 3+ years of React.js experience
        - Strong proficiency in JavaScript/TypeScript
        - Experience with CSS/SASS and responsive design
        - Understanding of RESTful APIs

        Nice-to-Have:
        - Experience with Next.js
        - Knowledge of GraphQL
        - Familiarity with testing libraries (Jest, RTL)
        """,
        "key_skills": ["React", "JavaScript", "TypeScript", "Redux", "Context API", "CSS", "SASS", "responsive design", "REST API", "Next.js", "GraphQL", "Jest", "unit testing"]
    },
    {
        "id": "JD003",
        "title": "Machine Learning Engineer",
        "company": "AI Innovations",
        "requirements": """
        Position: Machine Learning Engineer
        Location: On-site
        Employment Type: Full-time

        Key Responsibilities:
        - Develop and deploy ML models for production
        - Design feature engineering pipelines
        - Conduct A/B testing and model evaluation
        - Collaborate with data scientists on research projects

        Must-Have Requirements:
        - MS/PhD in Computer Science or related field
        - 3+ years of ML/Deep Learning experience
        - Proficiency in Python, TensorFlow/PyTorch
        - Experience with MLOps and model deployment

        Nice-to-Have:
        - Experience with NLP or Computer Vision
        - Knowledge of distributed computing (Spark)
        - Publications in top ML conferences
        """,
        "key_skills": ["Machine Learning", "Deep Learning", "Python", "TensorFlow", "PyTorch", "MLOps", "model deployment", "feature engineering", "A/B testing", "NLP", "Computer Vision", "Spark", "neural networks"]
    },
    {
        "id": "JD004",
        "title": "DevOps Engineer",
        "company": "CloudFirst Technologies",
        "requirements": """
        Position: DevOps Engineer
        Location: Remote
        Employment Type: Full-time

        Key Responsibilities:
        - Design and maintain CI/CD pipelines
        - Manage cloud infrastructure on AWS/Azure
        - Implement infrastructure as code using Terraform
        - Monitor system performance and troubleshoot issues

        Must-Have Requirements:
        - 4+ years of DevOps experience
        - Strong knowledge of Docker and Kubernetes
        - Experience with AWS or Azure services
        - Proficiency in scripting (Bash, Python)

        Nice-to-Have:
        - Experience with Ansible or Chef
        - Knowledge of monitoring tools (Prometheus, Grafana)
        - Security certifications
        """,
        "key_skills": ["DevOps", "CI/CD", "Docker", "Kubernetes", "AWS", "Azure", "Terraform", "infrastructure as code", "Bash", "Python", "Ansible", "Prometheus", "Grafana", "monitoring"]
    },
    {
        "id": "JD005",
        "title": "Full Stack JavaScript Developer",
        "company": "Digital Dynamics",
        "requirements": """
        Position: Full Stack JavaScript Developer
        Location: Hybrid
        Employment Type: Full-time

        Key Responsibilities:
        - Develop full-stack applications using Node.js and React
        - Design and implement database schemas
        - Build and maintain RESTful APIs
        - Ensure application security best practices

        Must-Have Requirements:
        - 4+ years of JavaScript experience
        - Proficiency in Node.js and Express.js
        - Experience with React or Vue.js
        - Knowledge of MongoDB or PostgreSQL

        Nice-to-Have:
        - Experience with TypeScript
        - Knowledge of WebSocket/real-time applications
        - Familiarity with AWS Lambda
        """,
        "key_skills": ["JavaScript", "Node.js", "Express.js", "React", "Vue.js", "MongoDB", "PostgreSQL", "REST API", "TypeScript", "WebSocket", "AWS Lambda", "full stack"]
    },
    {
        "id": "JD006",
        "title": "Data Scientist",
        "company": "Analytics Pro",
        "requirements": """
        Position: Data Scientist
        Location: On-site
        Employment Type: Full-time

        Key Responsibilities:
        - Analyze complex datasets to derive business insights
        - Build predictive models and recommendation systems
        - Create data visualizations and dashboards
        - Present findings to stakeholders

        Must-Have Requirements:
        - MS in Statistics, Mathematics, or related field
        - 3+ years of data science experience
        - Proficiency in Python, R, and SQL
        - Experience with statistical modeling

        Nice-to-Have:
        - Experience with big data tools (Hadoop, Spark)
        - Knowledge of cloud platforms
        - Domain expertise in finance or healthcare
        """,
        "key_skills": ["Data Science", "Python", "R", "SQL", "statistical modeling", "predictive models", "data visualization", "machine learning", "Hadoop", "Spark", "dashboards", "analytics"]
    },
    {
        "id": "JD007",
        "title": "iOS Mobile Developer",
        "company": "AppWorks Studio",
        "requirements": """
        Position: iOS Mobile Developer
        Location: Remote
        Employment Type: Full-time

        Key Responsibilities:
        - Develop and maintain iOS applications using Swift
        - Implement UI/UX designs following Apple guidelines
        - Integrate with backend APIs and third-party services
        - Ensure app performance and quality

        Must-Have Requirements:
        - 3+ years of iOS development experience
        - Strong proficiency in Swift and SwiftUI
        - Experience with Core Data and networking
        - Knowledge of App Store submission process

        Nice-to-Have:
        - Experience with Objective-C
        - Knowledge of CI/CD for mobile
        - Familiarity with augmented reality (ARKit)
        """,
        "key_skills": ["iOS", "Swift", "SwiftUI", "Core Data", "mobile development", "App Store", "Objective-C", "ARKit", "UIKit", "Xcode", "API integration"]
    },
    {
        "id": "JD008",
        "title": "Cybersecurity Analyst",
        "company": "SecureNet Solutions",
        "requirements": """
        Position: Cybersecurity Analyst
        Location: On-site
        Employment Type: Full-time

        Key Responsibilities:
        - Monitor security systems and investigate incidents
        - Conduct vulnerability assessments and penetration testing
        - Implement security policies and procedures
        - Provide security awareness training

        Must-Have Requirements:
        - 3+ years of cybersecurity experience
        - Knowledge of security frameworks (NIST, ISO 27001)
        - Experience with SIEM tools and threat detection
        - Understanding of network protocols and firewalls

        Nice-to-Have:
        - Security certifications (CISSP, CEH)
        - Experience with cloud security
        - Knowledge of forensics tools
        """,
        "key_skills": ["Cybersecurity", "security", "NIST", "ISO 27001", "SIEM", "threat detection", "vulnerability assessment", "penetration testing", "firewalls", "network security", "CISSP", "incident response"]
    },
    {
        "id": "JD009",
        "title": "Backend Java Developer",
        "company": "Enterprise Systems Ltd.",
        "requirements": """
        Position: Backend Java Developer
        Location: Hybrid
        Employment Type: Full-time

        Key Responsibilities:
        - Design and develop microservices using Spring Boot
        - Implement business logic and data processing
        - Optimize application performance and scalability
        - Participate in code reviews and architectural decisions

        Must-Have Requirements:
        - 5+ years of Java development experience
        - Strong knowledge of Spring Framework
        - Experience with SQL and NoSQL databases
        - Proficiency in RESTful API design

        Nice-to-Have:
        - Experience with Apache Kafka
        - Knowledge of containerization
        - Familiarity with event-driven architecture
        """,
        "key_skills": ["Java", "Spring Boot", "Spring Framework", "microservices", "SQL", "NoSQL", "REST API", "Kafka", "containerization", "backend", "JPA", "Hibernate"]
    },
    {
        "id": "JD010",
        "title": "Product Manager - Tech",
        "company": "InnovateTech",
        "requirements": """
        Position: Product Manager - Tech
        Location: Remote
        Employment Type: Full-time

        Key Responsibilities:
        - Define product vision and roadmap
        - Gather and prioritize requirements from stakeholders
        - Work with engineering teams to deliver features
        - Analyze product metrics and user feedback

        Must-Have Requirements:
        - 4+ years of product management experience
        - Strong understanding of software development
        - Experience with Agile/Scrum methodologies
        - Excellent communication and leadership skills

        Nice-to-Have:
        - Technical background (CS degree or similar)
        - Experience with data analytics tools
        - Domain expertise in SaaS products
        """,
        "key_skills": ["Product Management", "Agile", "Scrum", "roadmap", "stakeholder management", "software development", "user research", "product strategy", "SaaS", "analytics", "leadership"]
    },
    {
        "id": "JD011",
        "title": "Cloud Solutions Architect",
        "company": "CloudScale Systems",
        "requirements": """
        Position: Cloud Solutions Architect
        Location: Remote
        Employment Type: Full-time

        Key Responsibilities:
        - Design scalable cloud architectures on AWS/Azure/GCP
        - Lead cloud migration projects
        - Establish best practices for cloud security and cost optimization
        - Mentor engineering teams on cloud technologies

        Must-Have Requirements:
        - 6+ years of software engineering experience
        - 3+ years of cloud architecture experience
        - AWS/Azure/GCP certifications
        - Strong knowledge of networking and security

        Nice-to-Have:
        - Experience with multi-cloud strategies
        - Knowledge of serverless architectures
        - Experience with large-scale migrations
        """,
        "key_skills": ["Cloud Architecture", "AWS", "Azure", "GCP", "cloud migration", "security", "cost optimization", "serverless", "networking", "multi-cloud", "infrastructure", "scalability"]
    },
    {
        "id": "JD012",
        "title": "QA Automation Engineer",
        "company": "QualityFirst Software",
        "requirements": """
        Position: QA Automation Engineer
        Location: Hybrid
        Employment Type: Full-time

        Key Responsibilities:
        - Design and implement automated test frameworks
        - Create and maintain test scripts and documentation
        - Integrate tests into CI/CD pipelines
        - Collaborate with developers to improve code quality

        Must-Have Requirements:
        - 3+ years of QA automation experience
        - Proficiency in Selenium, Cypress, or Playwright
        - Experience with programming (Python/Java/JavaScript)
        - Knowledge of API testing tools (Postman, REST Assured)

        Nice-to-Have:
        - Experience with performance testing
        - Knowledge of mobile testing
        - ISTQB certification
        """,
        "key_skills": ["QA", "automation", "Selenium", "Cypress", "Playwright", "testing", "CI/CD", "API testing", "Postman", "test framework", "Python", "Java", "JavaScript"]
    },
    {
        "id": "JD013",
        "title": "Database Administrator",
        "company": "DataCore Solutions",
        "requirements": """
        Position: Database Administrator
        Location: On-site
        Employment Type: Full-time

        Key Responsibilities:
        - Manage and maintain database systems
        - Optimize database performance and queries
        - Implement backup, recovery, and security procedures
        - Plan and execute database migrations

        Must-Have Requirements:
        - 5+ years of DBA experience
        - Expert knowledge of PostgreSQL or MySQL
        - Experience with database replication and clustering
        - Strong SQL optimization skills

        Nice-to-Have:
        - Experience with cloud databases (RDS, Cloud SQL)
        - Knowledge of NoSQL databases
        - Oracle or SQL Server experience
        """,
        "key_skills": ["Database", "DBA", "PostgreSQL", "MySQL", "SQL", "replication", "clustering", "backup", "recovery", "performance tuning", "RDS", "database administration"]
    },
    {
        "id": "JD014",
        "title": "Blockchain Developer",
        "company": "Web3 Ventures",
        "requirements": """
        Position: Blockchain Developer
        Location: Remote
        Employment Type: Full-time

        Key Responsibilities:
        - Develop smart contracts on Ethereum/Solana
        - Build decentralized applications (dApps)
        - Audit and optimize existing smart contracts
        - Integrate blockchain solutions with existing systems

        Must-Have Requirements:
        - 2+ years of blockchain development experience
        - Proficiency in Solidity or Rust
        - Understanding of DeFi protocols and NFTs
        - Experience with Web3.js or Ethers.js

        Nice-to-Have:
        - Experience with Layer 2 solutions
        - Knowledge of zero-knowledge proofs
        - Smart contract security certifications
        """,
        "key_skills": ["Blockchain", "Ethereum", "Solana", "Solidity", "Rust", "smart contracts", "DeFi", "NFT", "Web3.js", "Ethers.js", "dApps", "decentralized"]
    },
    {
        "id": "JD015",
        "title": "Technical Writer",
        "company": "DocuTech Solutions",
        "requirements": """
        Position: Technical Writer
        Location: Remote
        Employment Type: Full-time

        Key Responsibilities:
        - Create and maintain technical documentation
        - Write API documentation and developer guides
        - Collaborate with engineering teams to understand features
        - Ensure documentation accuracy and consistency

        Must-Have Requirements:
        - 3+ years of technical writing experience
        - Experience with API documentation (OpenAPI, Swagger)
        - Proficiency in documentation tools (Confluence, GitBook)
        - Basic understanding of programming concepts

        Nice-to-Have:
        - Software development background
        - Experience with docs-as-code approaches
        - Knowledge of SEO for documentation
        """,
        "key_skills": ["Technical Writing", "documentation", "API documentation", "OpenAPI", "Swagger", "Confluence", "GitBook", "developer guides", "docs-as-code", "technical communication"]
    },
    {
        "id": "JD016",
        "title": "Site Reliability Engineer",
        "company": "ReliableOps",
        "requirements": """
        Position: Site Reliability Engineer
        Location: Remote
        Employment Type: Full-time

        Key Responsibilities:
        - Ensure system reliability and uptime
        - Implement observability and monitoring solutions
        - Automate operational tasks and incident response
        - Define and track SLOs/SLIs

        Must-Have Requirements:
        - 4+ years of SRE or DevOps experience
        - Strong knowledge of Linux systems
        - Experience with monitoring tools (Datadog, PagerDuty)
        - Proficiency in Python or Go

        Nice-to-Have:
        - Experience with chaos engineering
        - Knowledge of service mesh (Istio, Linkerd)
        - Background in distributed systems
        """,
        "key_skills": ["SRE", "Site Reliability", "Linux", "monitoring", "Datadog", "PagerDuty", "observability", "SLO", "SLI", "Python", "Go", "incident response", "automation"]
    },
    {
        "id": "JD017",
        "title": "UI/UX Designer",
        "company": "DesignHub Creative",
        "requirements": """
        Position: UI/UX Designer
        Location: Hybrid
        Employment Type: Full-time

        Key Responsibilities:
        - Create user-centered designs for web and mobile
        - Conduct user research and usability testing
        - Develop wireframes, prototypes, and design systems
        - Collaborate with developers on implementation

        Must-Have Requirements:
        - 3+ years of UI/UX design experience
        - Proficiency in Figma and design systems
        - Experience with user research methodologies
        - Strong portfolio demonstrating design process

        Nice-to-Have:
        - Experience with motion design
        - Knowledge of accessibility standards
        - Front-end development skills
        """,
        "key_skills": ["UI/UX", "design", "Figma", "user research", "usability testing", "wireframes", "prototypes", "design systems", "accessibility", "user experience", "visual design"]
    },
    {
        "id": "JD018",
        "title": "Embedded Systems Engineer",
        "company": "IoT Innovations",
        "requirements": """
        Position: Embedded Systems Engineer
        Location: On-site
        Employment Type: Full-time

        Key Responsibilities:
        - Design and develop firmware for embedded systems
        - Interface with sensors and peripherals
        - Optimize code for resource-constrained devices
        - Debug and troubleshoot hardware/software issues

        Must-Have Requirements:
        - 4+ years of embedded systems experience
        - Proficiency in C/C++ for embedded
        - Experience with microcontrollers (ARM, STM32)
        - Knowledge of communication protocols (I2C, SPI, UART)

        Nice-to-Have:
        - Experience with RTOS
        - Knowledge of wireless protocols (BLE, LoRa)
        - PCB design familiarity
        """,
        "key_skills": ["Embedded Systems", "firmware", "C", "C++", "microcontrollers", "ARM", "STM32", "I2C", "SPI", "UART", "RTOS", "BLE", "IoT"]
    },
    {
        "id": "JD019",
        "title": "Natural Language Processing Engineer",
        "company": "LangTech AI",
        "requirements": """
        Position: NLP Engineer
        Location: Remote
        Employment Type: Full-time

        Key Responsibilities:
        - Develop NLP models for text classification and generation
        - Fine-tune large language models (LLMs)
        - Build conversational AI systems
        - Evaluate model performance and iterate

        Must-Have Requirements:
        - MS/PhD in NLP or related field
        - 3+ years of NLP experience
        - Proficiency in Python and deep learning frameworks
        - Experience with transformers and LLMs

        Nice-to-Have:
        - Publications in NLP conferences
        - Experience with RAG systems
        - Knowledge of multi-lingual NLP
        """,
        "key_skills": ["NLP", "Natural Language Processing", "LLM", "transformers", "Python", "deep learning", "PyTorch", "TensorFlow", "text classification", "RAG", "conversational AI", "BERT", "GPT"]
    },
    {
        "id": "JD020",
        "title": "Technical Support Engineer",
        "company": "SupportPro Tech",
        "requirements": """
        Position: Technical Support Engineer
        Location: Hybrid
        Employment Type: Full-time

        Key Responsibilities:
        - Provide technical support to enterprise customers
        - Troubleshoot and resolve complex technical issues
        - Document solutions and create knowledge base articles
        - Escalate critical issues to engineering teams

        Must-Have Requirements:
        - 2+ years of technical support experience
        - Strong understanding of web technologies
        - Experience with ticketing systems (Zendesk, Jira)
        - Excellent problem-solving and communication skills

        Nice-to-Have:
        - Programming/scripting knowledge
        - Experience with API debugging
        - Familiarity with cloud platforms
        """,
        "key_skills": ["Technical Support", "troubleshooting", "customer support", "Zendesk", "Jira", "web technologies", "API", "debugging", "problem-solving", "documentation"]
    },
]

# ============================================================================
# CANDIDATE PROFILES (20 Candidates with Varying Qualities)
# ============================================================================

@dataclass
class CandidateProfile:
    id: str
    name: str
    quality: str  # "weak", "average", "strong", "exceptional"
    resume: str
    answer_style: str  # Description of how they answer
    target_jd_id: str  # Which JD they're applying for


CANDIDATES = [
    # WEAK CANDIDATES (5)
    CandidateProfile(
        id="C001",
        name="John Doe",
        quality="weak",
        target_jd_id="JD001",
        answer_style="Gives very short, vague answers. Often says 'I don't know' or repeats the question. Lacks technical depth.",
        resume="""
        John Doe
        Email: john.doe@email.com
        Phone: 555-0101

        Summary:
        Recent graduate looking for Python developer position.

        Education:
        - BS Computer Science, Local University (2023)

        Experience:
        - Intern at Small Company (3 months)
          - Helped with some Python scripts

        Skills:
        - Basic Python
        - HTML
        """
    ),
    CandidateProfile(
        id="C002",
        name="Jane Smith",
        quality="weak",
        target_jd_id="JD002",
        answer_style="Provides irrelevant answers. Doesn't understand technical concepts. Very nervous and unclear.",
        resume="""
        Jane Smith
        Email: jane.smith@email.com

        Summary:
        Looking for frontend developer role.

        Education:
        - Some college courses in web design

        Experience:
        - Personal blog using WordPress

        Skills:
        - WordPress
        - Basic HTML/CSS
        """
    ),
    CandidateProfile(
        id="C003",
        name="Bob Wilson",
        quality="weak",
        target_jd_id="JD003",
        answer_style="Makes incorrect technical statements. Overconfident but lacks actual knowledge.",
        resume="""
        Bob Wilson
        Email: bob.wilson@email.com

        Summary:
        Aspiring ML engineer with passion for AI.

        Education:
        - Online courses in machine learning

        Experience:
        - Completed some Kaggle tutorials

        Skills:
        - Python basics
        - Watched ML videos
        """
    ),
    CandidateProfile(
        id="C004",
        name="Alice Brown",
        quality="weak",
        target_jd_id="JD004",
        answer_style="Gives generic textbook answers without practical experience. Can't provide examples.",
        resume="""
        Alice Brown
        Email: alice.brown@email.com

        Summary:
        Seeking DevOps role.

        Education:
        - BS Information Technology (2022)

        Experience:
        - IT Help Desk (1 year)

        Skills:
        - Basic Linux commands
        - Read about Docker
        """
    ),
    CandidateProfile(
        id="C005",
        name="Charlie Davis",
        quality="weak",
        target_jd_id="JD005",
        answer_style="Extremely brief answers. Often just says 'yes' or 'no'. No elaboration.",
        resume="""
        Charlie Davis
        Email: charlie.d@email.com

        Summary:
        Full stack developer wannabe.

        Experience:
        - Built a todo app from tutorial

        Skills:
        - JavaScript basics
        - Following tutorials
        """
    ),

    # AVERAGE CANDIDATES (5)
    CandidateProfile(
        id="C006",
        name="Emily Johnson",
        quality="average",
        target_jd_id="JD006",
        answer_style="Provides adequate answers with some technical knowledge. Can explain concepts but lacks depth.",
        resume="""
        Emily Johnson
        Email: emily.johnson@email.com

        Summary:
        Data analyst transitioning to data science.

        Education:
        - MS Statistics, State University (2021)

        Experience:
        - Data Analyst at MidSize Corp (2 years)
          - Created reports using SQL and Excel
          - Built basic dashboards in Tableau
          - Some Python for data cleaning

        Skills:
        - Python, R, SQL
        - Tableau, Excel
        - Basic ML concepts
        """
    ),
    CandidateProfile(
        id="C007",
        name="Michael Chen",
        quality="average",
        target_jd_id="JD007",
        answer_style="Gives acceptable answers but struggles with follow-up questions. Knows basics well.",
        resume="""
        Michael Chen
        Email: m.chen@email.com

        Summary:
        iOS developer with 2 years experience.

        Education:
        - BS Computer Science (2021)

        Experience:
        - Junior iOS Developer at App Company (2 years)
          - Maintained existing iOS app
          - Fixed bugs and added minor features
          - Used Swift and UIKit

        Skills:
        - Swift, UIKit
        - Basic SwiftUI
        - REST API integration
        """
    ),
    CandidateProfile(
        id="C008",
        name="Sarah Williams",
        quality="average",
        target_jd_id="JD008",
        answer_style="Provides textbook-correct answers but limited real-world examples. Moderate depth.",
        resume="""
        Sarah Williams
        Email: sarah.w@email.com

        Summary:
        Cybersecurity professional with security focus.

        Education:
        - BS Cybersecurity (2020)
        - CompTIA Security+

        Experience:
        - Security Analyst at SmallTech (2 years)
          - Monitored security alerts
          - Participated in incident response
          - Performed basic vulnerability scans

        Skills:
        - Security monitoring
        - Vulnerability assessment
        - Basic penetration testing
        """
    ),
    CandidateProfile(
        id="C009",
        name="David Lee",
        quality="average",
        target_jd_id="JD009",
        answer_style="Solid basic knowledge. Can explain concepts clearly but misses advanced topics.",
        resume="""
        David Lee
        Email: david.lee@email.com

        Summary:
        Java developer with backend focus.

        Education:
        - BS Computer Science (2019)

        Experience:
        - Java Developer at Enterprise Corp (3 years)
          - Developed REST APIs using Spring Boot
          - Worked with MySQL databases
          - Participated in code reviews

        Skills:
        - Java, Spring Boot
        - MySQL, JPA
        - Git, Maven
        """
    ),
    CandidateProfile(
        id="C010",
        name="Lisa Garcia",
        quality="average",
        target_jd_id="JD010",
        answer_style="Good communication but technical depth varies. Provides reasonable examples.",
        resume="""
        Lisa Garcia
        Email: lisa.garcia@email.com

        Summary:
        Product manager with 3 years experience.

        Education:
        - MBA (2020)
        - BS Business Administration

        Experience:
        - Associate Product Manager at Tech Startup (3 years)
          - Managed product backlog
          - Conducted user interviews
          - Worked with engineering teams

        Skills:
        - Jira, Confluence
        - User research
        - Agile/Scrum
        """
    ),

    # STRONG CANDIDATES (5)
    CandidateProfile(
        id="C011",
        name="James Anderson",
        quality="strong",
        target_jd_id="JD011",
        answer_style="Provides detailed, comprehensive answers with specific examples. Strong technical depth.",
        resume="""
        James Anderson
        Email: james.anderson@email.com

        Summary:
        Senior Cloud Engineer with 7 years of experience in designing and implementing
        scalable cloud solutions. AWS Certified Solutions Architect Professional.

        Education:
        - MS Computer Science, Top University (2016)

        Experience:
        - Senior Cloud Architect at BigTech Inc (4 years)
          - Led migration of 50+ applications to AWS
          - Designed multi-region disaster recovery solutions
          - Reduced cloud costs by 40% through optimization
          - Mentored team of 5 engineers

        - Cloud Engineer at Tech Corp (3 years)
          - Built CI/CD pipelines using Jenkins and AWS
          - Implemented Infrastructure as Code with Terraform
          - Managed Kubernetes clusters at scale

        Skills:
        - AWS, Azure, GCP
        - Terraform, CloudFormation
        - Docker, Kubernetes
        - Python, Go

        Certifications:
        - AWS Solutions Architect Professional
        - AWS DevOps Engineer Professional
        - CKA (Certified Kubernetes Administrator)
        """
    ),
    CandidateProfile(
        id="C012",
        name="Jennifer Martinez",
        quality="strong",
        target_jd_id="JD012",
        answer_style="Clear, structured answers with practical examples. Demonstrates expertise through detailed explanations.",
        resume="""
        Jennifer Martinez
        Email: j.martinez@email.com

        Summary:
        QA Automation Lead with 5 years of experience building test frameworks
        and improving software quality processes.

        Education:
        - BS Computer Engineering (2018)
        - ISTQB Advanced Level

        Experience:
        - QA Automation Lead at Software Corp (3 years)
          - Built Selenium framework reducing manual testing by 70%
          - Implemented API testing with REST Assured
          - Integrated tests into Jenkins CI/CD pipeline
          - Led team of 4 QA engineers

        - QA Engineer at Tech Startup (2 years)
          - Developed automated tests using Cypress
          - Created performance testing strategy
          - Improved bug detection rate by 50%

        Skills:
        - Selenium, Cypress, Playwright
        - Java, Python, JavaScript
        - REST Assured, Postman
        - Jenkins, GitHub Actions
        """
    ),
    CandidateProfile(
        id="C013",
        name="Robert Taylor",
        quality="strong",
        target_jd_id="JD013",
        answer_style="Expert-level responses with deep technical knowledge. Provides architecture-level insights.",
        resume="""
        Robert Taylor
        Email: r.taylor@email.com

        Summary:
        Senior Database Administrator with 8 years experience managing enterprise
        database systems and leading data platform initiatives.

        Education:
        - MS Database Systems (2015)

        Experience:
        - Senior DBA at Enterprise Solutions (5 years)
          - Managed PostgreSQL clusters handling 10TB+ data
          - Designed high-availability database architecture
          - Reduced query times by 60% through optimization
          - Led database migration to cloud (RDS)

        - Database Administrator at FinTech Corp (3 years)
          - Administered MySQL and PostgreSQL databases
          - Implemented replication and backup strategies
          - Created monitoring and alerting systems

        Skills:
        - PostgreSQL, MySQL
        - AWS RDS, Aurora
        - Performance tuning
        - Backup/Recovery
        - High Availability
        """
    ),
    CandidateProfile(
        id="C014",
        name="Michelle Wang",
        quality="strong",
        target_jd_id="JD014",
        answer_style="Articulate with strong technical foundation. Explains complex concepts clearly with examples.",
        resume="""
        Michelle Wang
        Email: michelle.wang@email.com

        Summary:
        Blockchain Developer with 3 years of experience building decentralized
        applications and smart contracts.

        Education:
        - MS Computer Science, Blockchain specialization (2020)

        Experience:
        - Senior Blockchain Developer at DeFi Protocol (2 years)
          - Developed smart contracts handling $50M+ TVL
          - Built DEX aggregator increasing trade efficiency by 30%
          - Led security audit process
          - Contributed to open-source Ethereum tools

        - Blockchain Developer at Web3 Startup (1 year)
          - Created NFT marketplace smart contracts
          - Integrated with multiple EVM chains
          - Built frontend with React and ethers.js

        Skills:
        - Solidity, Vyper
        - Web3.js, Ethers.js
        - Hardhat, Foundry
        - React, TypeScript
        """
    ),
    CandidateProfile(
        id="C015",
        name="Kevin Brown",
        quality="strong",
        target_jd_id="JD015",
        answer_style="Excellent communication. Provides well-organized, thorough answers with relevant context.",
        resume="""
        Kevin Brown
        Email: kevin.brown@email.com

        Summary:
        Senior Technical Writer with 6 years of experience creating developer
        documentation and API guides.

        Education:
        - BA English, Technical Communication minor (2017)
        - Technical Writing certification

        Experience:
        - Senior Technical Writer at API Platform (4 years)
          - Created API documentation used by 10,000+ developers
          - Built docs-as-code pipeline with GitHub
          - Reduced support tickets by 40% through better docs
          - Led documentation team of 3 writers

        - Technical Writer at SaaS Company (2 years)
          - Wrote user guides and tutorials
          - Created video documentation
          - Collaborated with engineering on release notes

        Skills:
        - OpenAPI/Swagger
        - Markdown, YAML
        - Git, GitHub
        - Docs-as-code tools
        - API documentation
        """
    ),

    # EXCEPTIONAL CANDIDATES (5)
    CandidateProfile(
        id="C016",
        name="Dr. Amanda Foster",
        quality="exceptional",
        target_jd_id="JD019",
        answer_style="Expert-level, nuanced answers. Cites research and provides innovative perspectives. Exceptional depth.",
        resume="""
        Dr. Amanda Foster
        Email: amanda.foster@email.com

        Summary:
        NLP Research Scientist with 8 years of experience in natural language
        processing and deep learning. Published researcher with industry impact.

        Education:
        - PhD Computer Science (NLP), Stanford University (2019)
        - MS Artificial Intelligence, MIT (2015)

        Experience:
        - Senior NLP Scientist at AI Research Lab (4 years)
          - Led team developing LLM fine-tuning framework
          - Published 5 papers in ACL/EMNLP conferences
          - Built RAG system serving 1M+ queries/day
          - Advised on company NLP strategy

        - Research Scientist at Tech Giant (3 years)
          - Developed BERT-based text classification models
          - Created conversational AI for customer service
          - Filed 3 patents in NLP technology

        Skills:
        - PyTorch, TensorFlow, JAX
        - Transformers, LLMs
        - Python, C++
        - Distributed training

        Publications:
        - 12 papers in top NLP conferences
        - 2,000+ citations
        """
    ),
    CandidateProfile(
        id="C017",
        name="Alex Thompson",
        quality="exceptional",
        target_jd_id="JD016",
        answer_style="Provides comprehensive, strategic answers. Demonstrates leadership and technical excellence.",
        resume="""
        Alex Thompson
        Email: alex.thompson@email.com

        Summary:
        Principal SRE with 10 years of experience building and scaling reliable
        systems at top tech companies. Led teams through hypergrowth.

        Education:
        - MS Computer Science, UC Berkeley (2014)

        Experience:
        - Principal SRE at Scale-Up Unicorn (4 years)
          - Built SRE team from 2 to 15 engineers
          - Achieved 99.99% uptime for critical services
          - Reduced incident response time by 60%
          - Implemented chaos engineering program

        - Senior SRE at FAANG Company (4 years)
          - Managed infrastructure for 100M+ users
          - Designed observability platform
          - Created SLO framework adopted company-wide

        - Systems Engineer at Startup (2 years)
          - Built initial infrastructure from scratch
          - Implemented CI/CD and monitoring

        Skills:
        - Kubernetes, Docker
        - Prometheus, Grafana, Datadog
        - Python, Go
        - Terraform, Ansible
        - Distributed Systems
        """
    ),
    CandidateProfile(
        id="C018",
        name="Sophia Rodriguez",
        quality="exceptional",
        target_jd_id="JD017",
        answer_style="Creative, insightful answers. Strong portfolio backing. Articulates design thinking excellently.",
        resume="""
        Sophia Rodriguez
        Email: sophia.r@email.com

        Summary:
        Lead Product Designer with 8 years of experience creating user-centered
        designs for Fortune 500 companies and successful startups.

        Education:
        - MFA Interaction Design, Parsons (2016)
        - BA Graphic Design

        Experience:
        - Lead Product Designer at Design-Forward Startup (4 years)
          - Redesigned product increasing engagement by 45%
          - Built design system used by 50+ designers
          - Led user research program
          - Managed team of 6 designers

        - Senior UX Designer at Enterprise Corp (3 years)
          - Designed B2B SaaS platform
          - Conducted 100+ user interviews
          - Created accessibility guidelines

        Awards:
        - Webby Award for Best UX (2022)
        - Fast Company Innovation by Design (2021)

        Skills:
        - Figma, Sketch, Adobe XD
        - User Research
        - Design Systems
        - Prototyping
        - Accessibility (WCAG)
        """
    ),
    CandidateProfile(
        id="C019",
        name="Daniel Kim",
        quality="exceptional",
        target_jd_id="JD018",
        answer_style="Deep technical expertise. Explains complex embedded concepts with clarity. Strong problem-solving.",
        resume="""
        Daniel Kim
        Email: daniel.kim@email.com

        Summary:
        Principal Embedded Systems Engineer with 12 years of experience developing
        firmware for IoT and industrial applications.

        Education:
        - MS Electrical Engineering, Georgia Tech (2012)

        Experience:
        - Principal Engineer at IoT Leader (6 years)
          - Architected firmware for smart home platform
          - Reduced power consumption by 50%
          - Led team of 8 embedded engineers
          - Managed product from concept to 1M+ devices shipped

        - Senior Embedded Engineer at Industrial Corp (4 years)
          - Developed real-time control systems
          - Implemented safety-critical firmware (IEC 61508)
          - Created automated testing framework

        - Embedded Developer at Startup (2 years)
          - Built BLE-enabled wearable device
          - Optimized battery life to 6 months

        Skills:
        - C, C++, Assembly
        - ARM, STM32, ESP32
        - FreeRTOS, Zephyr
        - BLE, LoRa, WiFi
        - PCB design basics

        Patents:
        - 4 patents in IoT technology
        """
    ),
    CandidateProfile(
        id="C020",
        name="Rachel Green",
        quality="exceptional",
        target_jd_id="JD020",
        answer_style="Outstanding communication. Demonstrates empathy and technical skills. Provides comprehensive solutions.",
        resume="""
        Rachel Green
        Email: rachel.green@email.com

        Summary:
        Senior Technical Support Engineer with 7 years of experience supporting
        enterprise customers and building support excellence programs.

        Education:
        - BS Information Systems (2017)
        - AWS Solutions Architect Associate

        Experience:
        - Senior Support Engineer at Enterprise SaaS (4 years)
          - Maintained 98% customer satisfaction rating
          - Resolved 500+ complex technical issues
          - Built knowledge base reducing ticket volume by 35%
          - Trained team of 10 support engineers

        - Technical Support Specialist at Tech Company (3 years)
          - Provided API integration support
          - Created troubleshooting documentation
          - Participated in on-call rotation

        Skills:
        - API debugging
        - Cloud platforms (AWS, GCP)
        - Python, JavaScript
        - SQL
        - Zendesk, Jira
        - Customer communication
        """
    ),
]

# ============================================================================
# ANSWER GENERATORS (Based on Candidate Quality)
# ============================================================================

def generate_candidate_answer(candidate: CandidateProfile, question: str, question_type: str) -> str:
    """
    Generate a candidate answer based on their quality profile.
    Uses LLM to simulate realistic responses.
    """
    # Check if this is a closing question
    if "questions for me" in question.lower() or "wrap up" in question.lower():
        return "No, thank you. I think you've covered everything I wanted to know."

    quality_prompts = {
        "weak": """
You are simulating a WEAK candidate in a job interview.
Characteristics:
- Give very short, vague answers (1-2 sentences)
- Sometimes say "I don't know" or "I'm not sure"
- Lack technical depth
- May misunderstand the question
- Cannot provide specific examples
- Sometimes repeat the question instead of answering
""",
        "average": """
You are simulating an AVERAGE candidate in a job interview.
Characteristics:
- Give adequate answers with moderate detail (3-4 sentences)
- Show basic understanding of concepts
- Provide some examples but not very detailed
- Miss some advanced aspects
- Generally correct but not comprehensive
""",
        "strong": """
You are simulating a STRONG candidate in a job interview.
Characteristics:
- Give detailed, comprehensive answers (4-6 sentences)
- Demonstrate solid technical knowledge
- Provide specific examples from experience
- Show problem-solving approach
- Cover most aspects of the question well
""",
        "exceptional": """
You are simulating an EXCEPTIONAL candidate in a job interview.
Characteristics:
- Give excellent, nuanced answers (5-7 sentences)
- Demonstrate expert-level knowledge
- Provide detailed examples with measurable outcomes
- Show strategic thinking and innovation
- Address edge cases and best practices
- Connect answers to broader context
"""
    }

    prompt = f"""{quality_prompts[candidate.quality]}

Candidate Background:
{candidate.resume}

The interviewer asked: "{question}"

This is a {question_type} question.

Provide a realistic answer from this candidate's perspective.
Keep the answer natural and conversational.
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "I'm not entirely sure about that."

# ============================================================================
# TEST RESULT DATA STRUCTURES
# ============================================================================

@dataclass
class TechnicalQA:
    """Represents a technical question-answer pair with analysis"""
    question: str
    answer: str
    question_type: str
    relevance_score: float
    relevance_justification: str = ""
    matched_skills: List[str] = field(default_factory=list)
    candidate_struggled: bool = False
    ai_provided_explanation: bool = False
    explanation_text: str = ""

@dataclass
class InterviewQuestion:
    """Represents a question asked during interview"""
    turn: int
    question_text: str
    question_type: str  # "introduction", "project", "technical", "follow_up", "closing"
    tool_used: str  # Which tool was called before this question
    relevance_score: float = 0.0
    relevance_justification: str = ""
    matched_skills: List[str] = field(default_factory=list)

@dataclass
class InterviewResult:
    """Complete result of one interview"""
    candidate_id: str
    candidate_name: str
    candidate_quality: str
    jd_id: str
    jd_title: str

    # Interview flow
    questions: List[InterviewQuestion] = field(default_factory=list)
    total_turns: int = 0
    interview_duration_turns: int = 0

    # Technical Q&A tracking
    technical_qas: List[TechnicalQA] = field(default_factory=list)

    # Raw outputs
    full_transcript: str = ""
    evaluation_report: str = ""
    hr_report: str = ""

    # Analysis metrics
    question_relevance_scores: List[float] = field(default_factory=list)
    avg_relevance_score: float = 0.0
    recommendation: str = ""

    # Evaluation scores
    evaluation_scores: List[int] = field(default_factory=list)
    avg_evaluation_score: float = 0.0

    # Errors
    errors: List[str] = field(default_factory=list)

@dataclass
class TestSummary:
    """Summary of all test results"""
    timestamp: str
    total_interviews: int
    successful_interviews: int
    failed_interviews: int

    # Flow consistency metrics
    flow_patterns: Dict[str, int] = field(default_factory=dict)  # Pattern -> count
    avg_questions_per_interview: float = 0.0
    question_count_variance: float = 0.0

    # Question relevance metrics (semantic similarity)
    avg_relevance_score: float = 0.0
    relevance_by_position: Dict[str, float] = field(default_factory=dict)
    relevance_by_quality: Dict[str, float] = field(default_factory=dict)

    # Evaluation score metrics
    avg_eval_score_by_quality: Dict[str, float] = field(default_factory=dict)

    # Recommendation distribution
    recommendation_by_quality: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Report structure analysis
    report_sections_consistency: Dict[str, float] = field(default_factory=dict)

    # Detailed results
    results: List[InterviewResult] = field(default_factory=list)

# ============================================================================
# LLM-BASED RELEVANCE AUDITOR (LLM-as-a-Judge)
# ============================================================================

# Initialize the relevance auditor LLM
relevance_auditor_llm = init_chat_model("google_genai:gemini-2.5-flash-lite", temperature=0.1)

RELEVANCE_AUDITOR_PROMPT = """You are a Quality Assurance (QA) Auditor specializing in evaluating the relevance of interview questions to job descriptions.

Your task is to evaluate whether a TECHNICAL interview question is relevant to the given job description.

## Job Description:
{job_description}

## Technical Question to Evaluate:
{question}

## Evaluation Criteria:
- Score 5 (Highly Relevant): The question directly tests a skill, technology, or concept explicitly mentioned in the JD or closely related to its requirements.
- Score 4 (Relevant): The question tests a skill that is related to the job requirements, even if not explicitly mentioned (e.g., asking about "PyTorch" for a JD requiring "Deep Learning").
- Score 3 (Somewhat Relevant): The question tests general technical skills that could be useful for the role but are not specifically aligned with the JD.
- Score 2 (Marginally Relevant): The question has weak connection to the job requirements.
- Score 1 (Not Relevant): The question is unrelated to the job description requirements.

## Response Format (JSON):
{{
    "relevance_score": <1-5>,
    "justification": "<brief explanation of why this score was given>",
    "matched_skills": ["<list of JD skills/requirements this question tests>"]
}}

Respond ONLY with the JSON object, no additional text."""


def evaluate_question_relevance_llm(question: str, jd: Dict) -> Tuple[float, str, List[str]]:
    """
    Use LLM-as-a-Judge to evaluate the relevance of a technical question to the JD.
    Returns: (normalized_score 0-1, justification, matched_skills)
    """
    try:
        prompt = RELEVANCE_AUDITOR_PROMPT.format(
            job_description=jd.get("requirements", ""),
            question=question
        )

        response = relevance_auditor_llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()

        # Parse JSON response
        # Clean up response if it has markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)

        score = result.get("relevance_score", 3)
        justification = result.get("justification", "No justification provided")
        matched_skills = result.get("matched_skills", [])

        # Normalize score to 0-1 range (1-5 -> 0.2-1.0)
        normalized_score = score / 5.0

        return normalized_score, justification, matched_skills

    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Response was: {response_text[:200]}...")
        return 0.6, "Error parsing response", []
    except Exception as e:
        print(f"Error in LLM relevance evaluation: {e}")
        return 0.6, f"Error: {str(e)}", []


def batch_evaluate_relevance(questions: List[str], jd: Dict) -> List[Dict]:
    """
    Evaluate multiple technical questions for relevance to JD.
    Returns list of evaluation results.
    """
    results = []
    for question in questions:
        score, justification, matched_skills = evaluate_question_relevance_llm(question, jd)
        results.append({
            "question": question,
            "relevance_score": score,
            "justification": justification,
            "matched_skills": matched_skills
        })
    return results

# ============================================================================
# INTERVIEW SIMULATOR
# ============================================================================

class InterviewSimulator:
    """Simulates interviews and collects test data"""

    def __init__(self):
        self.workflow = build_workflow()
        self.results: List[InterviewResult] = []

    def run_interview(self, candidate: CandidateProfile, jd: Dict) -> InterviewResult:
        """Run a single interview and collect results"""
        result = InterviewResult(
            candidate_id=candidate.id,
            candidate_name=candidate.name,
            candidate_quality=candidate.quality,
            jd_id=jd["id"],
            jd_title=jd["title"]
        )

        print(f"\n{'='*60}")
        print(f"Starting Interview: {candidate.name} ({candidate.quality}) for {jd['title']}")
        print(f"{'='*60}")

        try:
            # Initialize state
            state = AgentState(
                mode="technical",
                num_of_q=3,
                num_of_follow_up=2,
                position=jd["title"],
                company_name=jd["company"],
                messages=[],
                resume_text=candidate.resume,
                resume_path=None,
                questions_path=None,
                pdf_path=None,
                evaluation_result="",
                report=""
            )

            # Save JD temporarily for the interview
            self._save_temp_jd(jd)

            # Run the interview conversation
            transcript_lines = []
            turn = 0
            max_turns = 20  # Safety limit
            interview_ended = False
            last_ai_question = ""
            last_candidate_answer = ""

            # Start conversation
            state["messages"] = [HumanMessage(content="Start the interview")]

            while turn < max_turns and not interview_ended:
                turn += 1
                print(f"\n--- Turn {turn} ---")

                # Get AI response
                try:
                    updated_state = self.workflow.invoke(state)
                    state = updated_state
                except Exception as e:
                    print(f"Workflow error: {e}")
                    result.errors.append(f"Turn {turn}: {str(e)}")
                    break

                # Find the last AI message
                ai_message = None
                for msg in reversed(state["messages"]):
                    if isinstance(msg, AIMessage) and msg.content:
                        ai_message = msg
                        break

                if not ai_message or not ai_message.content:
                    print("No AI response received")
                    continue

                ai_text = ai_message.content.strip()
                print(f"AI: {ai_text[:200]}...")
                transcript_lines.append(f"AI Recruiter: {ai_text}")

                # Check if interview ended
                if "that's it for today" in ai_text.lower():
                    interview_ended = True
                    print("Interview ended.")
                    break

                # Analyze the question
                question_info = self._analyze_question(ai_text, turn, state, jd)
                result.questions.append(question_info)

                # Check if AI provided explanation (for weak candidates)
                ai_provided_explanation = False
                explanation_text = ""
                if last_candidate_answer and candidate.quality == "weak":
                    # Check if AI is explaining after a struggle
                    explanation_indicators = [
                        "let me explain", "to clarify", "in other words",
                        "what this means", "essentially", "the answer is",
                        "you're on the right track", "to help you understand"
                    ]
                    if any(ind in ai_text.lower() for ind in explanation_indicators):
                        ai_provided_explanation = True
                        explanation_text = ai_text

                # Generate candidate response
                candidate_answer = generate_candidate_answer(
                    candidate,
                    ai_text,
                    question_info.question_type
                )
                print(f"Candidate: {candidate_answer[:200]}...")
                transcript_lines.append(f"Candidate: {candidate_answer}")

                # Track technical Q&A
                if question_info.question_type in ["technical", "project"]:
                    struggled = False
                    if candidate.quality == "weak":
                        struggle_indicators = ["not sure", "don't know", "i think", "maybe", "uh"]
                        struggled = any(ind in candidate_answer.lower() for ind in struggle_indicators)

                    qa = TechnicalQA(
                        question=ai_text,
                        answer=candidate_answer,
                        question_type=question_info.question_type,
                        relevance_score=question_info.relevance_score,
                        candidate_struggled=struggled,
                        ai_provided_explanation=ai_provided_explanation,
                        explanation_text=explanation_text
                    )
                    result.technical_qas.append(qa)

                last_ai_question = ai_text
                last_candidate_answer = candidate_answer

                # Add to state
                state["messages"] = list(state["messages"]) + [HumanMessage(content=candidate_answer)]

            result.total_turns = turn
            result.interview_duration_turns = len([q for q in result.questions if q.question_type != "closing"])
            result.full_transcript = "\n\n".join(transcript_lines)

            # Calculate average relevance score
            if result.questions:
                tech_questions = [q for q in result.questions if q.question_type in ["technical", "project"]]
                if tech_questions:
                    result.question_relevance_scores = [q.relevance_score for q in tech_questions]
                    result.avg_relevance_score = sum(result.question_relevance_scores) / len(result.question_relevance_scores)

            # Run evaluation
            print("\nRunning evaluation...")
            try:
                # Manually invoke evaluator
                from src.dynamic_workflow import evaluator, report_writer

                eval_result = evaluator(state)
                state["evaluation_result"] = eval_result.get("evaluation_result", "")
                result.evaluation_report = state["evaluation_result"]
                print(f"Evaluation: {result.evaluation_report[:300]}...")

                # Extract evaluation scores
                result.evaluation_scores = self._extract_evaluation_scores(result.evaluation_report)
                if result.evaluation_scores:
                    result.avg_evaluation_score = sum(result.evaluation_scores) / len(result.evaluation_scores)

                # Run report writer
                report_result = report_writer(state)
                state["report"] = report_result.get("report", "")
                result.hr_report = state["report"]
                print(f"Report: {result.hr_report[:300]}...")

                # Extract recommendation (full text, no truncation)
                result.recommendation = self._extract_recommendation(result.hr_report)

            except Exception as e:
                print(f"Evaluation error: {e}")
                traceback.print_exc()
                result.errors.append(f"Evaluation: {str(e)}")

        except Exception as e:
            print(f"Interview error: {e}")
            traceback.print_exc()
            result.errors.append(f"General: {str(e)}")

        return result

    def _save_temp_jd(self, jd: Dict):
        """Save JD temporarily for interview"""
        os.makedirs("Improved Job Descriptions", exist_ok=True)
        jd_path = "Improved Job Descriptions/Improved_Job_Description.pdf"

        # Create a simple text file (the system will still try to load it)
        # For testing, we'll create the JD as a document
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet

        doc = SimpleDocTemplate(jd_path, pagesize=letter)
        styles = getSampleStyleSheet()

        content = [
            Paragraph(f"<b>{jd['title']}</b>", styles['Heading1']),
            Paragraph(f"Company: {jd['company']}", styles['Normal']),
            Paragraph(jd['requirements'].replace('\n', '<br/>'), styles['Normal'])
        ]

        doc.build(content)

    def _analyze_question(self, question_text: str, turn: int, state: Dict, jd: Dict) -> InterviewQuestion:
        """Analyze and categorize a question"""
        question_type = "technical"
        tool_used = ""

        # Check recent tool calls
        for msg in reversed(state.get("messages", [])[-5:]):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_used = msg.tool_calls[0].get("name", "")
                break

        # Categorize question
        text_lower = question_text.lower()
        if turn == 1 or "tell me about yourself" in text_lower or "introduce yourself" in text_lower:
            question_type = "introduction"
        elif "project" in text_lower or "experience" in text_lower or tool_used == "retrieve_resume":
            question_type = "project"
        elif "questions for me" in text_lower or "wrap up" in text_lower:
            question_type = "closing"
        elif "could you elaborate" in text_lower or "can you explain" in text_lower:
            question_type = "follow_up"
        else:
            question_type = "technical"

        # Calculate relevance score using LLM-as-a-Judge ONLY for technical questions
        # IMPORTANT: Only measure relevance for technical questions, NOT project/introduction
        relevance_score = 0.0
        relevance_justification = ""
        matched_skills = []

        if question_type == "technical":
            # Use LLM-based relevance auditor for technical questions only
            relevance_score, relevance_justification, matched_skills = evaluate_question_relevance_llm(
                question_text, jd
            )

        return InterviewQuestion(
            turn=turn,
            question_text=question_text,
            question_type=question_type,
            tool_used=tool_used,
            relevance_score=relevance_score,
            relevance_justification=relevance_justification,
            matched_skills=matched_skills
        )

    def _extract_recommendation(self, report: str) -> str:
        """Extract the FULL recommendation from HR report (no truncation)"""
        # Look for recommendation section
        patterns = [
            r"Overall Recommendation[:\s]*\n*(.*?)(?=\n\n|\n###|\Z)",
            r"### Overall Recommendation[:\s]*\n*(.*?)(?=\n\n|\n###|\Z)",
        ]

        for pattern in patterns:
            match = re.search(pattern, report, re.IGNORECASE | re.DOTALL)
            if match:
                recommendation = match.group(1).strip()
                # Clean up but don't truncate
                recommendation = re.sub(r'\n+', ' ', recommendation)
                return recommendation

        # Fallback: look for key phrases
        if "proceed to next round" in report.lower():
            return "Proceed to next round"
        elif "not a fit" in report.lower():
            return "Not a fit at this time"
        elif "consider with reservations" in report.lower():
            return "Consider with reservations"

        return "Recommendation not found"

    def _extract_evaluation_scores(self, evaluation: str) -> List[int]:
        """Extract numerical scores from evaluation report"""
        scores = []
        # Look for patterns like "Score: X/5" or "X/5"
        pattern = r'Score:\s*(\d)/5'
        matches = re.findall(pattern, evaluation, re.IGNORECASE)
        for match in matches:
            try:
                scores.append(int(match))
            except:
                pass
        return scores

# ============================================================================
# TEST ANALYSIS FUNCTIONS
# ============================================================================

def analyze_flow_consistency(results: List[InterviewResult]) -> Dict:
    """Analyze if interview flow is consistent across interviews"""
    analysis = {
        "flow_patterns": defaultdict(int),
        "question_counts": [],
        "question_type_sequences": [],
        "questions_by_quality": defaultdict(list),
    }

    for result in results:
        if result.errors:
            continue

        # Create flow pattern string
        types = [q.question_type for q in result.questions]
        pattern = "->".join(types[:5])  # First 5 question types
        analysis["flow_patterns"][pattern] += 1

        # Count questions
        analysis["question_counts"].append(len(result.questions))
        analysis["question_type_sequences"].append(types)
        analysis["questions_by_quality"][result.candidate_quality].append(len(result.questions))

    # Calculate statistics
    counts = analysis["question_counts"]
    if counts:
        analysis["avg_questions"] = sum(counts) / len(counts)
        analysis["min_questions"] = min(counts)
        analysis["max_questions"] = max(counts)
        mean = analysis["avg_questions"]
        analysis["variance"] = sum((x - mean) ** 2 for x in counts) / len(counts)
        analysis["std_dev"] = analysis["variance"] ** 0.5

    # Average questions by quality
    for quality, q_counts in analysis["questions_by_quality"].items():
        analysis[f"avg_questions_{quality}"] = sum(q_counts) / len(q_counts) if q_counts else 0

    # Check if most common pattern is dominant
    if analysis["flow_patterns"]:
        most_common = max(analysis["flow_patterns"].values())
        total = sum(analysis["flow_patterns"].values())
        analysis["flow_consistency_score"] = most_common / total

    return dict(analysis)

def analyze_question_relevance(results: List[InterviewResult]) -> Dict:
    """
    Analyze how relevant TECHNICAL questions are to job descriptions using LLM-as-a-Judge.
    IMPORTANT: Only measures relevance for technical questions, NOT project/introduction questions.
    """
    analysis = {
        "overall_avg_relevance": 0.0,
        "relevance_by_position": defaultdict(list),
        "relevance_by_quality": defaultdict(list),
        "all_scores": [],
        "total_technical_questions": 0,
        "evaluations_with_justification": [],  # Store LLM justifications
        "matched_skills_by_position": defaultdict(list),  # Track which skills were tested
    }

    for result in results:
        # Only count technical questions
        technical_questions = [q for q in result.questions if q.question_type == "technical"]
        analysis["total_technical_questions"] += len(technical_questions)

        for q in technical_questions:
            if q.relevance_score > 0:
                analysis["all_scores"].append(q.relevance_score)
                analysis["relevance_by_position"][result.jd_title].append(q.relevance_score)
                analysis["relevance_by_quality"][result.candidate_quality].append(q.relevance_score)

                # Store LLM evaluation details
                if q.relevance_justification:
                    analysis["evaluations_with_justification"].append({
                        "position": result.jd_title,
                        "question": q.question_text[:100] + "..." if len(q.question_text) > 100 else q.question_text,
                        "score": q.relevance_score,
                        "justification": q.relevance_justification,
                        "matched_skills": q.matched_skills
                    })

                # Track matched skills
                if q.matched_skills:
                    analysis["matched_skills_by_position"][result.jd_title].extend(q.matched_skills)

    if analysis["all_scores"]:
        analysis["overall_avg_relevance"] = sum(analysis["all_scores"]) / len(analysis["all_scores"])
        analysis["min_relevance"] = min(analysis["all_scores"])
        analysis["max_relevance"] = max(analysis["all_scores"])
        mean = analysis["overall_avg_relevance"]
        analysis["std_dev"] = (sum((x - mean) ** 2 for x in analysis["all_scores"]) / len(analysis["all_scores"])) ** 0.5

    # Calculate averages per position
    for pos, scores in list(analysis["relevance_by_position"].items()):
        analysis["relevance_by_position"][pos] = sum(scores) / len(scores) if scores else 0.0

    for quality, scores in list(analysis["relevance_by_quality"].items()):
        analysis["relevance_by_quality"][quality] = sum(scores) / len(scores) if scores else 0.0

    # Deduplicate matched skills per position
    for pos, skills in analysis["matched_skills_by_position"].items():
        analysis["matched_skills_by_position"][pos] = list(set(skills))

    return dict(analysis)

def analyze_evaluation_scores(results: List[InterviewResult]) -> Dict:
    """Analyze evaluation scores by candidate quality"""
    analysis = {
        "scores_by_quality": defaultdict(list),
        "avg_score_by_quality": {},
        "all_scores": [],
    }

    for result in results:
        if result.evaluation_scores:
            analysis["all_scores"].extend(result.evaluation_scores)
            analysis["scores_by_quality"][result.candidate_quality].extend(result.evaluation_scores)

    # Calculate averages
    for quality, scores in analysis["scores_by_quality"].items():
        analysis["avg_score_by_quality"][quality] = sum(scores) / len(scores) if scores else 0

    if analysis["all_scores"]:
        analysis["overall_avg"] = sum(analysis["all_scores"]) / len(analysis["all_scores"])

    return dict(analysis)

def analyze_question_difficulty(results: List[InterviewResult]) -> Dict:
    """Analyze if question difficulty is consistent"""
    analysis = {
        "technical_question_lengths": [],
        "complexity_indicators": defaultdict(list),
        "struggles_by_quality": defaultdict(int),
        "explanations_provided": defaultdict(int),
    }

    # Indicators of question complexity
    complexity_keywords = [
        "explain", "describe", "compare", "analyze", "design",
        "implement", "optimize", "how would you", "what if"
    ]

    for result in results:
        for q in result.questions:
            if q.question_type == "technical":
                analysis["technical_question_lengths"].append(len(q.question_text.split()))

                # Count complexity indicators
                text_lower = q.question_text.lower()
                count = sum(1 for kw in complexity_keywords if kw in text_lower)
                analysis["complexity_indicators"][result.jd_title].append(count)

        # Track struggles and explanations
        for qa in result.technical_qas:
            if qa.candidate_struggled:
                analysis["struggles_by_quality"][result.candidate_quality] += 1
            if qa.ai_provided_explanation:
                analysis["explanations_provided"][result.candidate_quality] += 1

    # Calculate statistics
    lengths = analysis["technical_question_lengths"]
    if lengths:
        analysis["avg_question_length"] = sum(lengths) / len(lengths)
        analysis["length_variance"] = sum((x - analysis["avg_question_length"]) ** 2 for x in lengths) / len(lengths)

    # Average complexity per position
    for pos, counts in analysis["complexity_indicators"].items():
        analysis["complexity_indicators"][pos] = sum(counts) / len(counts) if counts else 0

    return dict(analysis)

def analyze_recommendation_variation(results: List[InterviewResult]) -> Dict:
    """Analyze if recommendations vary appropriately based on candidate quality"""
    analysis = {
        "recommendations_by_quality": defaultdict(lambda: defaultdict(int)),
        "expected_alignment": 0.0,
        "recommendation_categories": defaultdict(lambda: defaultdict(int)),
    }

    # Define expected recommendations
    expected_positive = {"proceed", "next round", "recommend", "strong candidate", "highly suitable"}
    expected_negative = {"not a fit", "not suitable", "not recommend", "does not meet"}
    expected_mixed = {"reservations", "consider with", "potential", "further assessment"}

    correct_alignments = 0
    total = 0

    for result in results:
        if result.recommendation and result.recommendation != "Recommendation not found":
            rec_lower = result.recommendation.lower()
            analysis["recommendations_by_quality"][result.candidate_quality][result.recommendation] += 1

            # Categorize recommendation
            if any(pos in rec_lower for pos in expected_positive):
                category = "Positive"
            elif any(neg in rec_lower for neg in expected_negative):
                category = "Negative"
            elif any(mix in rec_lower for mix in expected_mixed):
                category = "Mixed"
            else:
                category = "Other"

            analysis["recommendation_categories"][result.candidate_quality][category] += 1

            # Check alignment
            is_positive = category == "Positive"
            is_negative = category == "Negative"
            is_mixed = category == "Mixed"

            if result.candidate_quality in ["strong", "exceptional"]:
                if is_positive:
                    correct_alignments += 1
            elif result.candidate_quality == "weak":
                if is_negative:
                    correct_alignments += 1
            else:  # average
                if is_mixed or is_positive:
                    correct_alignments += 0.75  # Average can go either way

            total += 1

    if total > 0:
        analysis["expected_alignment"] = correct_alignments / total

    return dict(analysis)

def analyze_report_structure(results: List[InterviewResult]) -> Dict:
    """Analyze if report structure is consistent"""
    expected_sections = [
        "Candidate Overall Suitability",
        "Key Strengths",
        "Areas for Development",
        "Technical Skills Demonstrated",
        "Communication Effectiveness",
        "Overall Recommendation"
    ]

    analysis = {
        "section_presence": defaultdict(int),
        "section_consistency": {},
        "total_reports": 0,
    }

    for result in results:
        if result.hr_report:
            analysis["total_reports"] += 1
            report_lower = result.hr_report.lower()

            for section in expected_sections:
                if section.lower() in report_lower or section.replace(" ", "").lower() in report_lower:
                    analysis["section_presence"][section] += 1

    # Calculate consistency percentages
    if analysis["total_reports"] > 0:
        for section in expected_sections:
            analysis["section_consistency"][section] = (
                analysis["section_presence"][section] / analysis["total_reports"]
            )

    return dict(analysis)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def generate_visualizations(results: List[InterviewResult], summary: TestSummary, output_dir: str):
    """Generate all visualizations for the report"""
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Relevance Score Distribution Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    all_relevance = []
    for result in results:
        all_relevance.extend(result.question_relevance_scores)

    if all_relevance:
        ax.hist(all_relevance, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(x=np.mean(all_relevance), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_relevance):.3f}')
        ax.set_xlabel('Semantic Relevance Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Question Relevance Scores (Semantic Similarity)', fontsize=14)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'relevance_histogram.png'), dpi=150)
        plt.close()

    # 2. Relevance by Position (Bar Chart)
    fig, ax = plt.subplots(figsize=(14, 8))
    positions = list(summary.relevance_by_position.keys())
    scores = [summary.relevance_by_position[p] for p in positions]

    colors = ['green' if s >= 0.7 else 'orange' if s >= 0.5 else 'red' for s in scores]
    bars = ax.barh(positions, scores, color=colors, edgecolor='black')
    ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.7, label='Good (0.7)')
    ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Acceptable (0.5)')
    ax.set_xlabel('Semantic Relevance Score', fontsize=12)
    ax.set_title('Question Relevance by Position', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'relevance_by_position.png'), dpi=150)
    plt.close()

    # 3. Evaluation Scores by Candidate Quality (Box Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    quality_order = ['weak', 'average', 'strong', 'exceptional']
    scores_by_quality = {q: [] for q in quality_order}

    for result in results:
        if result.evaluation_scores:
            scores_by_quality[result.candidate_quality].extend(result.evaluation_scores)

    data = [scores_by_quality[q] for q in quality_order]
    bp = ax.boxplot(data, labels=[q.capitalize() for q in quality_order], patch_artist=True)

    colors_box = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)

    ax.set_xlabel('Candidate Quality', fontsize=12)
    ax.set_ylabel('Evaluation Score (1-5)', fontsize=12)
    ax.set_title('Evaluation Scores Distribution by Candidate Quality', fontsize=14)
    ax.set_ylim(0, 6)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'scores_by_quality_boxplot.png'), dpi=150)
    plt.close()

    # 4. Recommendation Distribution (Stacked Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    qualities = ['weak', 'average', 'strong', 'exceptional']
    categories = ['Negative', 'Mixed', 'Positive', 'Other']
    colors_rec = ['#ff6b6b', '#ffd93d', '#6bcb77', '#aaaaaa']

    rec_analysis = analyze_recommendation_variation(results)

    bottoms = [0] * len(qualities)
    for i, cat in enumerate(categories):
        values = [rec_analysis["recommendation_categories"].get(q, {}).get(cat, 0) for q in qualities]
        ax.bar([q.capitalize() for q in qualities], values, bottom=bottoms, label=cat, color=colors_rec[i])
        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_xlabel('Candidate Quality', fontsize=12)
    ax.set_ylabel('Number of Recommendations', fontsize=12)
    ax.set_title('Recommendation Distribution by Candidate Quality', fontsize=14)
    ax.legend(title='Recommendation Type')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'recommendation_distribution.png'), dpi=150)
    plt.close()

    # 5. Questions per Interview by Quality (Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    flow_analysis = analyze_flow_consistency(results)

    avg_q = [flow_analysis.get(f"avg_questions_{q}", 0) for q in quality_order]
    ax.bar([q.capitalize() for q in quality_order], avg_q, color=['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff'], edgecolor='black')
    ax.set_xlabel('Candidate Quality', fontsize=12)
    ax.set_ylabel('Average Number of Questions', fontsize=12)
    ax.set_title('Average Interview Length by Candidate Quality', fontsize=14)

    # Add value labels
    for i, v in enumerate(avg_q):
        ax.text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'questions_by_quality.png'), dpi=150)
    plt.close()

    # 6. Heatmap: Relevance by Position and Question Type
    fig, ax = plt.subplots(figsize=(12, 10))

    # Collect data
    positions = list(set(r.jd_title for r in results))
    q_types = ['technical', 'project']

    heatmap_data = np.zeros((len(positions), len(q_types)))
    counts = np.zeros((len(positions), len(q_types)))

    for result in results:
        pos_idx = positions.index(result.jd_title)
        for q in result.questions:
            if q.question_type in q_types:
                type_idx = q_types.index(q.question_type)
                heatmap_data[pos_idx, type_idx] += q.relevance_score
                counts[pos_idx, type_idx] += 1

    # Calculate averages
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_data = np.where(counts > 0, heatmap_data / counts, 0)

    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(q_types)))
    ax.set_xticklabels([t.capitalize() for t in q_types], fontsize=11)
    ax.set_yticks(range(len(positions)))
    ax.set_yticklabels(positions, fontsize=9)

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Relevance Score', fontsize=12)

    # Add text annotations
    for i in range(len(positions)):
        for j in range(len(q_types)):
            text = f'{heatmap_data[i, j]:.2f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=8)

    ax.set_title('Question Relevance Heatmap by Position and Type', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'relevance_heatmap.png'), dpi=150)
    plt.close()

    # 7. Report Section Consistency (Horizontal Bar)
    fig, ax = plt.subplots(figsize=(10, 6))
    sections = list(summary.report_sections_consistency.keys())
    consistency = [summary.report_sections_consistency[s] * 100 for s in sections]

    colors_sec = ['green' if c == 100 else 'orange' if c >= 80 else 'red' for c in consistency]
    ax.barh(sections, consistency, color=colors_sec, edgecolor='black')
    ax.axvline(x=100, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Consistency Rate (%)', fontsize=12)
    ax.set_title('HR Report Section Consistency', fontsize=14)
    ax.set_xlim(0, 110)

    for i, v in enumerate(consistency):
        ax.text(v + 1, i, f'{v:.0f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'report_consistency.png'), dpi=150)
    plt.close()

    # 8. Summary Metrics Radar Chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    categories = ['Flow\nConsistency', 'Question\nRelevance', 'Score\nAlignment',
                  'Report\nStructure', 'Recommendation\nAccuracy']

    flow_analysis = analyze_flow_consistency(results)
    rec_analysis = analyze_recommendation_variation(results)

    values = [
        flow_analysis.get("flow_consistency_score", 0) * 100,
        summary.avg_relevance_score * 100,
        (summary.avg_eval_score_by_quality.get('strong', 0) / 5) * 100 if summary.avg_eval_score_by_quality else 50,
        np.mean(list(summary.report_sections_consistency.values())) * 100 if summary.report_sections_consistency else 0,
        rec_analysis.get("expected_alignment", 0) * 100
    ]

    # Close the radar
    values += values[:1]
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]

    ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
    ax.fill(angles, values, alpha=0.25, color='steelblue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title('Overall System Performance Metrics', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'performance_radar.png'), dpi=150)
    plt.close()

    print(f"Visualizations saved to: {viz_dir}/")
    return viz_dir

# ============================================================================
# OUTPUT SAVING FUNCTIONS
# ============================================================================

def save_outputs(results: List[InterviewResult], summary: TestSummary):
    """Save all test outputs to files"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save individual interview transcripts with detailed Q&A
    transcripts_dir = os.path.join(OUTPUT_DIR, f"transcripts_{timestamp}")
    os.makedirs(transcripts_dir, exist_ok=True)

    for result in results:
        filename = f"{result.candidate_id}_{result.candidate_name.replace(' ', '_')}.txt"
        filepath = os.path.join(transcripts_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Interview Transcript\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"Candidate: {result.candidate_name}\n")
            f.write(f"Quality Profile: {result.candidate_quality}\n")
            f.write(f"Position: {result.jd_title}\n")
            f.write(f"Total Turns: {result.total_turns}\n")
            f.write(f"Average Relevance Score: {result.avg_relevance_score:.3f}\n")
            f.write(f"Average Evaluation Score: {result.avg_evaluation_score:.2f}/5\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(result.full_transcript)

            # Add Technical Q&A Section
            f.write(f"\n\n{'=' * 60}\n")
            f.write("TECHNICAL QUESTIONS AND ANSWERS\n")
            f.write(f"{'=' * 60}\n\n")

            for i, qa in enumerate(result.technical_qas, 1):
                f.write(f"Q{i} [{qa.question_type.upper()}] (Relevance: {qa.relevance_score:.3f})\n")
                f.write(f"Question: {qa.question}\n\n")
                f.write(f"Answer: {qa.answer}\n")
                if qa.candidate_struggled:
                    f.write(f"[CANDIDATE STRUGGLED]\n")
                if qa.ai_provided_explanation:
                    f.write(f"[AI PROVIDED EXPLANATION]: {qa.explanation_text[:200]}...\n")
                f.write(f"\n{'-' * 40}\n\n")

            f.write(f"\n{'=' * 60}\n")
            f.write("EVALUATION REPORT\n")
            f.write(f"{'=' * 60}\n")
            f.write(result.evaluation_report)
            f.write(f"\n\n{'=' * 60}\n")
            f.write("HR REPORT\n")
            f.write(f"{'=' * 60}\n")
            f.write(result.hr_report)
            f.write(f"\n\n{'=' * 60}\n")
            f.write("RECOMMENDATION (FULL)\n")
            f.write(f"{'=' * 60}\n")
            f.write(result.recommendation)

    # 2. Save summary JSON
    summary_path = os.path.join(OUTPUT_DIR, f"test_summary_{timestamp}.json")

    # Convert summary to serializable dict
    summary_dict = {
        "timestamp": summary.timestamp,
        "total_interviews": summary.total_interviews,
        "successful_interviews": summary.successful_interviews,
        "failed_interviews": summary.failed_interviews,
        "flow_patterns": dict(summary.flow_patterns),
        "avg_questions_per_interview": summary.avg_questions_per_interview,
        "question_count_variance": summary.question_count_variance,
        "avg_relevance_score": summary.avg_relevance_score,
        "relevance_by_position": dict(summary.relevance_by_position),
        "relevance_by_quality": dict(summary.relevance_by_quality),
        "avg_eval_score_by_quality": dict(summary.avg_eval_score_by_quality),
        "recommendation_by_quality": {k: dict(v) for k, v in summary.recommendation_by_quality.items()},
        "report_sections_consistency": dict(summary.report_sections_consistency),
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, indent=2)

    # 3. Save detailed analysis report
    analysis_path = os.path.join(OUTPUT_DIR, f"detailed_analysis_{timestamp}.txt")
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("4MSHIRE AI - INTERVIEW STAGE TEST AND EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test Date: {timestamp}\n")
        f.write(f"Total Interviews Conducted: {summary.total_interviews}\n")
        f.write(f"Successful: {summary.successful_interviews}\n")
        f.write(f"Failed: {summary.failed_interviews}\n\n")

        f.write("-" * 80 + "\n")
        f.write("1. INTERVIEW FLOW CONSISTENCY ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average Questions per Interview: {summary.avg_questions_per_interview:.2f}\n")
        f.write(f"Question Count Variance: {summary.question_count_variance:.2f}\n")
        f.write(f"\nFlow Patterns Observed:\n")
        for pattern, count in summary.flow_patterns.items():
            f.write(f"  {pattern}: {count} occurrences\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("2. QUESTION RELEVANCE ANALYSIS (SEMANTIC SIMILARITY)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Overall Average Relevance Score: {summary.avg_relevance_score:.3f}\n")
        f.write(f"\nRelevance by Position:\n")
        for pos, score in sorted(summary.relevance_by_position.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {pos}: {score:.3f}\n")
        f.write(f"\nRelevance by Candidate Quality:\n")
        for quality, score in summary.relevance_by_quality.items():
            f.write(f"  {quality}: {score:.3f}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("3. EVALUATION SCORES BY QUALITY\n")
        f.write("-" * 80 + "\n")
        for quality, score in summary.avg_eval_score_by_quality.items():
            f.write(f"  {quality}: {score:.2f}/5\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("4. RECOMMENDATION VARIATION ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write("Recommendations by Candidate Quality:\n")
        for quality, recs in summary.recommendation_by_quality.items():
            f.write(f"\n  {quality.upper()} candidates:\n")
            for rec, count in recs.items():
                f.write(f"    - {rec}: {count}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("5. REPORT STRUCTURE CONSISTENCY\n")
        f.write("-" * 80 + "\n")
        f.write("Section Presence Rate:\n")
        for section, rate in summary.report_sections_consistency.items():
            f.write(f"  {section}: {rate*100:.1f}%\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("6. INDIVIDUAL INTERVIEW SUMMARIES\n")
        f.write("-" * 80 + "\n")
        for result in results:
            f.write(f"\n{result.candidate_name} ({result.candidate_quality}) - {result.jd_title}\n")
            f.write(f"  Questions: {len(result.questions)}\n")
            f.write(f"  Turns: {result.total_turns}\n")
            f.write(f"  Avg Relevance: {result.avg_relevance_score:.3f}\n")
            f.write(f"  Avg Eval Score: {result.avg_evaluation_score:.2f}/5\n")
            f.write(f"  Recommendation: {result.recommendation}\n")
            if result.errors:
                f.write(f"  Errors: {result.errors}\n")

    # 4. Save Technical Q&A Examples
    qa_examples_path = os.path.join(OUTPUT_DIR, f"technical_qa_examples_{timestamp}.txt")
    with open(qa_examples_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TECHNICAL QUESTIONS AND ANSWERS - EXAMPLES BY QUALITY\n")
        f.write("=" * 80 + "\n\n")

        for quality in ["weak", "average", "strong", "exceptional"]:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"{quality.upper()} CANDIDATES\n")
            f.write(f"{'=' * 60}\n\n")

            quality_results = [r for r in results if r.candidate_quality == quality]

            for result in quality_results[:2]:  # Show 2 examples per quality
                f.write(f"\nCandidate: {result.candidate_name} - {result.jd_title}\n")
                f.write(f"-" * 40 + "\n")

                for i, qa in enumerate(result.technical_qas[:3], 1):  # Show 3 Q&As
                    f.write(f"\nQ{i}: {qa.question}\n")
                    f.write(f"\nA{i}: {qa.answer}\n")
                    f.write(f"\nRelevance Score: {qa.relevance_score:.3f}")
                    if qa.candidate_struggled:
                        f.write(" [STRUGGLED]")
                    if qa.ai_provided_explanation:
                        f.write(f"\n[AI Explanation Provided]: {qa.explanation_text[:300]}...")
                    f.write(f"\n{'~' * 30}\n")

    # 5. Generate visualizations
    viz_dir = generate_visualizations(results, summary, OUTPUT_DIR)

    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print(f"  - Transcripts: {transcripts_dir}/")
    print(f"  - Summary JSON: {summary_path}")
    print(f"  - Analysis Report: {analysis_path}")
    print(f"  - Q&A Examples: {qa_examples_path}")
    print(f"  - Visualizations: {viz_dir}/")

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_tests(num_interviews: int = 20):
    """Run the full test suite"""
    print("=" * 80)
    print("4MSHIRE AI - INTERVIEW STAGE TESTING")
    print("=" * 80)
    print(f"Starting {num_interviews} interview simulations...")

    simulator = InterviewSimulator()
    results = []

    # Pair candidates with JDs (they're already matched by target_jd_id)
    test_pairs = []
    for candidate in CANDIDATES[:num_interviews]:
        jd = next((j for j in JOB_DESCRIPTIONS if j["id"] == candidate.target_jd_id), JOB_DESCRIPTIONS[0])
        test_pairs.append((candidate, jd))

    # Run interviews
    for i, (candidate, jd) in enumerate(test_pairs, 1):
        print(f"\n[{i}/{num_interviews}] Running interview...")
        result = simulator.run_interview(candidate, jd)
        results.append(result)

    # Analyze results
    print("\n" + "=" * 80)
    print("ANALYZING RESULTS...")
    print("=" * 80)

    flow_analysis = analyze_flow_consistency(results)
    relevance_analysis = analyze_question_relevance(results)
    eval_analysis = analyze_evaluation_scores(results)
    difficulty_analysis = analyze_question_difficulty(results)
    recommendation_analysis = analyze_recommendation_variation(results)
    report_analysis = analyze_report_structure(results)

    # Create summary
    summary = TestSummary(
        timestamp=datetime.now().isoformat(),
        total_interviews=num_interviews,
        successful_interviews=len([r for r in results if not r.errors]),
        failed_interviews=len([r for r in results if r.errors]),
        flow_patterns=dict(flow_analysis.get("flow_patterns", {})),
        avg_questions_per_interview=flow_analysis.get("avg_questions", 0),
        question_count_variance=flow_analysis.get("variance", 0),
        avg_relevance_score=relevance_analysis.get("overall_avg_relevance", 0),
        relevance_by_position=dict(relevance_analysis.get("relevance_by_position", {})),
        relevance_by_quality=dict(relevance_analysis.get("relevance_by_quality", {})),
        avg_eval_score_by_quality=dict(eval_analysis.get("avg_score_by_quality", {})),
        recommendation_by_quality=dict(recommendation_analysis.get("recommendations_by_quality", {})),
        report_sections_consistency=dict(report_analysis.get("section_consistency", {})),
        results=results
    )

    # Save outputs
    save_outputs(results, summary)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Interviews: {summary.total_interviews}")
    print(f"Successful: {summary.successful_interviews}")
    print(f"Failed: {summary.failed_interviews}")
    print(f"\nFlow Consistency Score: {flow_analysis.get('flow_consistency_score', 0):.2f}")
    print(f"Average Relevance Score (Semantic): {summary.avg_relevance_score:.3f}")
    print(f"Report Structure Consistency: {sum(summary.report_sections_consistency.values())/len(summary.report_sections_consistency) if summary.report_sections_consistency else 0:.2f}")

    print(f"\nEvaluation Scores by Quality:")
    for quality, score in summary.avg_eval_score_by_quality.items():
        print(f"  {quality}: {score:.2f}/5")

    print(f"\nRecommendation Alignment: {recommendation_analysis.get('expected_alignment', 0)*100:.1f}%")

    return summary

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test 4MSHire AI Interview Stage")
    parser.add_argument("--num", type=int, default=20, help="Number of interviews to run")
    args = parser.parse_args()

    summary = run_tests(num_interviews=args.num)

    print("\n" + "=" * 80)
    print("TESTING COMPLETE!")
    print("=" * 80)
