/**
 * @license
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {Component, Inject, Input} from '@angular/core';
import {MAT_DIALOG_DATA, MatDialogRef} from '@angular/material/dialog';
import {AgentRunRequest} from '../../core/models/AgentRunRequest';
import {AgentService} from '../../core/services/agent.service';

@Component({
  selector: 'app-pending-event-dialog',
  standalone: false,
  templateUrl: './pending-event-dialog.component.html',
  styleUrl: './pending-event-dialog.component.scss',
})
export class PendingEventDialogComponent {
  selectedEvent: any = null;
  appName: string;
  userId: string;
  sessionId: string;
  functionCallEventId: string;
  sending: boolean = false;

  constructor(
    public dialogRef: MatDialogRef<PendingEventDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: any,
    private agentService: AgentService,
  ) {
    this.selectedEvent = data.event;
    this.appName = data.appName;
    this.userId = data.userId;
    this.sessionId = data.sessionId;
    this.functionCallEventId = data.functionCallEventId;
  }

  argsToJson(args: any) {
    return JSON.stringify(args);
  }

  sendResponse() {
    this.sending = true;
    const req: AgentRunRequest = {
      appName: this.appName,
      userId: this.userId,
      sessionId: this.sessionId,
      newMessage: {
        'role': 'user',
        'parts': [],
      },
    };
    if (this.selectedEvent.response) {
      req.functionCallEventId = this.functionCallEventId;
      req.newMessage.parts.push({
        'function_response': {
          id: this.selectedEvent.id,
          name: this.selectedEvent.name,
          response: {'response': this.selectedEvent.response},
        },
      });
    }
    this.agentService.run(req).subscribe((res) => {
      this.sending = false;
      for (const e of res) {
        if (e.content.parts[0].text) {
          this.dialogRef.close({
            text: e.content.parts[0].text,
            events: [this.selectedEvent],
          });
        }
      }
    });
  }
}
