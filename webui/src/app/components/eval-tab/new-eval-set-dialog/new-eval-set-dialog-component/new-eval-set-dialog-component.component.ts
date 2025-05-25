/**
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

import {Component, EventEmitter, Inject, Output} from '@angular/core';
import {MAT_DIALOG_DATA, MatDialogRef} from '@angular/material/dialog';
import {v4 as uuidv4} from 'uuid';
import {EvalService} from '../../../../core/services/eval.service';

@Component({
  selector: 'app-new-eval-set-dialog-component',
  templateUrl: './new-eval-set-dialog-component.component.html',
  styleUrl: './new-eval-set-dialog-component.component.scss',
  standalone: false,
})
export class NewEvalSetDialogComponentComponent {
  newSetId: string = 'evalset' + uuidv4().slice(0, 6);

  constructor(
    private evalService: EvalService,
    @Inject(MAT_DIALOG_DATA) public data: {appName: string},
    public dialogRef: MatDialogRef<NewEvalSetDialogComponentComponent>,
  ) {}

  createNewEvalSet() {
    if (!this.newSetId || this.newSetId == '') {
      alert('Cannot create eval set with empty id!');
    } else {
      this.evalService
        .createNewEvalSet(this.data.appName, this.newSetId)
        .subscribe((res) => {
          this.dialogRef.close(true);
        });
    }
  }
}
