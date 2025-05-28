import { Component, EventEmitter, Output } from '@angular/core';

@Component({
  selector: 'app-starter-questions',
  templateUrl: './starter-questions.component.html',
  styleUrls: ['./starter-questions.component.scss'],
  standalone: false,
})
export class StarterQuestionsComponent {
  @Output() questionSelected = new EventEmitter<string>();

  // Define your static starter questions here
  starterQuestions: string[] = [
    "What is Google ADK?",
    "How do I set up a new ADK agent?",
    "Explain the AgentTool in ADK.",
    "Can you help with GitHub issue #123 in google/adk-python?",
    "Show in a sequence diagram how the Runner orchestrates the entire lifecycle of an agent interaction for a given user session.",
    "Create a PDF document about ADK tools."
  ];

  constructor() { }

  onQuestionClick(question: string): void {
    this.questionSelected.emit(question);
  }
}