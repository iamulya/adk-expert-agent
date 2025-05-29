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
    "Write unit tests for code_executor_context.py",
    "Explain in detail how a user query is processed in the Google ADK.",
    "Can you help with GitHub issue #123?",
    "Show in a sequence diagram how the Runner orchestrates the entire lifecycle of an agent interaction for a given user session.",
    "Create a PDF document about ADK tools."
  ];

  constructor() { }

  onQuestionClick(question: string): void {
    this.questionSelected.emit(question);
  }
}